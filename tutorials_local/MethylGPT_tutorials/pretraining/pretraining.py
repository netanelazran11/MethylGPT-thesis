import argparse
import ast
import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch # Ensure torch is imported
from tqdm import tqdm

from methylgpt.model.methyl_datasets import create_dataloader
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab
from methylgpt.model.methyl_loss import masked_mse_loss
from methylgpt.utils.logging import * # Assuming this imports setup_logger, add_console_handler
# from methylgpt.utils.plot_embeddings import plot_umap_categorical, plot_umap_numerical # Not used in current main loop
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value # Ensure this is imported

from utils import * # This was in the original, ensure its contents are available or explicitly imported

try:
    from torch.cuda.amp import GradScaler
    amp_available = True
except ImportError:
    amp_available = False
    warnings.warn("torch.cuda.amp.GradScaler not available. AMP will be disabled if configured.")

try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Moved to bash script
# os.environ['TORCH_USE_CUDA_DSA'] = "1" # Moved to bash script

# Create parser for optional command-line overrides
parser = argparse.ArgumentParser()
parser.add_argument("-config_file", "--config_file", help="Path to the config file", default="config.json")
parser.add_argument("-device", "--device", help="Device to use for training (e.g., cuda:0 or cpu)", default="cuda:0")
parser.add_argument("-savename", "--savename", help="Name for saving outputs", default="pretraining_test")
parser.add_argument("-probe_id_path", "--probe_id_path", help="Path to probe IDs CSV file for VOCABULARY creation", default="data/probe_ids_type3.csv")
parser.add_argument("-parquet_data_dir", "--parquet_data_dir", help="Path to PREPROCESSED parquet data directory for TRAINING", default="preprocessed_data/parquet_files")
parser.add_argument("-metadata_file", "--metadata_file", help="Path to the PREPROCESSED QCed_samples_type3.csv metadata file for TRAINING", default="preprocessed_data/QCed_samples_type3.csv")

try:
    args = parser.parse_args()
except:
    args = argparse.Namespace()
    args.config_file = "config.json"
    args.device = "cuda:0"
    args.savename = "pretraining_test"
    args.probe_id_path = "data/probe_ids_type3.csv" # For vocab
    args.parquet_data_dir = "preprocessed_data/parquet_files" # For training data
    args.metadata_file = "preprocessed_data/QCed_samples_type3.csv" # For training metadata

# Load config from JSON file
with open(args.config_file, 'r') as f:
    config_from_file = json.load(f)

# Initialize W&B (if not already initialized by environment variables)
# The environment variables WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME set in the shell script
# should be automatically picked up by wandb.init().
# You can override or set additional wandb parameters here if needed.

if config_from_file.get("do_train", True): # Only init wandb if training
    try:
        import wandb
        # Check if WANDB_PROJECT is set, otherwise wandb.init() might prompt or error
        if os.getenv("WANDB_PROJECT"):
            wandb.init(
                project=os.getenv("WANDB_PROJECT"), 
                entity=os.getenv("WANDB_ENTITY"), # Optional, will use default if not set
                name=os.getenv("WANDB_NAME"),     # Optional, will be auto-generated if not set
                config=config_from_file # Log the initial config from file
            )
            print(f"Weights & Biases initialized for project: {os.getenv('WANDB_PROJECT')}, run: {wandb.run.name}")
        else:
            print("WANDB_PROJECT environment variable not set. Skipping wandb.init().")
            wandb = None # Disable wandb logging if project not set
    except ImportError:
        print("wandb library not found, skipping Weights & Biases initialization.")
        wandb = None # Ensure wandb is None if import fails
    except Exception as e:
        print(f"Error initializing Weights & Biases: {e}")
        wandb = None # Ensure wandb is None if init fails
else:
    wandb = None # Disable wandb if not training


# Initialize MethylVocab early to get its length for max_seq_len
# Determine save_dir for MethylVocab. This needs to be consistent with later usage.
# We construct a preliminary save_dir path here. This assumes savename and data_type are from args.
prelim_savename = args.savename
# prelim_data_type = args.data_type # Removed
# prelim_dataset_name = f"CpGs_type{prelim_data_type}" # Removed
# Ensure save_dir for vocab is defined. For now, let's use a temporary path or ensure it's set.
# This is a bit tricky as save_dir depends on time, which hasn't been set yet for the final save_dir.
# For vocab saving, it might be okay if it's saved to a temp location or if vocab saving is conditional.
# Let's assume probe_id_dir, pad_token, and special_tokens are available in config_from_file or have defaults.

probe_id_path = Path(args.probe_id_path) # Use args.probe_id_path
pad_token_val = config_from_file.get("pad_token", "<pad>")
special_tokens_val = config_from_file.get("special_tokens", ["<pad>", "<cls>", "<eoc>"])

# Temporary save_dir for vocab, or None if saving is handled later/conditionally
# We need to define save_dir for MethylVocab. Let's define it similar to how it's done later.
# However, the timestamp will be different if we define it here.
# A cleaner way might be to pass the vocab object to the model and other components, 
# and calculate max_seq_len just before model instantiation if it depends on vocab size.

# For now, let's instantiate vocab without a save_dir or a fixed one if it's only for length.
# The MethylVocab class saves the vocab if save_dir is not None.
# To avoid premature saving to a time-stamped dir, pass None or a fixed path.
methyl_vocab_instance = MethylVocab(probe_id_path, pad_token_val, special_tokens_val, save_dir=None) # Pass None for save_dir initially

config = dict(
    # Important thing to control
    seed=config_from_file.get("seed", 42),
    # input_type=f"CpGs_type{args.data_type}", # Removed
    parquet_dir=Path(args.parquet_data_dir), # USE PREPROCESSED PARQUET DIR
    probe_id_dir=probe_id_path, # This is for vocab, uses the original probe_id_path arg
    data_dir=Path(args.metadata_file),  # USE PREPROCESSED METADATA FILE
    valid_ratio=config_from_file.get("valid_ratio", 0.1),
    # n_hvg=n_hvg_predefined[config_from_file.get('data_type', '3')], # Removed n_hvg
    max_fi=config_from_file.get("max_fi", 500000),  # To use full dataset, Just set >500000
    do_train=config_from_file.get("do_train", True),
    pretrained_file=config_from_file.get("pretrained_file", None),  # None for pretraining from scratch
    mask_ratio=config_from_file.get("mask_ratio", 0.3),
    GEPC=config_from_file.get("GEPC", True),  # Masked value prediction for cell embedding
    dab_weight=config_from_file.get("dab_weight", 1.0),
    # pretraining_dataset_name=f"CpGs_type{args.data_type}", # Removed

    # Model and training
    epochs=config_from_file.get("epochs", 100),
    ecs_thres=config_from_file.get("ecs_thres", 0.0),  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    lr=config_from_file.get("lr", 1e-3),
    batch_size=config_from_file.get("batch_size", 32),  #4,  
    layer_size=config_from_file.get("layer_size", 64), #16,
    nlayers=config_from_file.get("nlayers", 6), #4,
    nhead=config_from_file.get("nhead", 4),
    dropout=config_from_file.get("dropout", 0.1),
    schedule_ratio=config_from_file.get("schedule_ratio", 0.9),  # ratio of epochs for learning rate schedule
    save_eval_interval=config_from_file.get("save_eval_interval", 10),
    log_interval=config_from_file.get("log_interval", 1000),
    fast_transformer=config_from_file.get("fast_transformer", True) and flash_attn_available, # Ensure flash_attn is available
    pre_norm=config_from_file.get("pre_norm", False),
    amp=config_from_file.get("amp", True) and amp_available, # Ensure amp is available


   # Additional tokens and values
    pad_token=pad_token_val, # Use the variable
    special_tokens=special_tokens_val, # Use the variable
    mask_value=config_from_file.get("mask_value", -1), # Define mask_value from config or default
    pad_value=config_from_file.get("pad_value", -2),    # Define pad_value from config or default
    explicit_zero_prob=config_from_file.get("explicit_zero_prob", False),  # Flag for explicit zero probability
    max_seq_len=len(methyl_vocab_instance.CpG_list) + 1,  # Use length of CpG list from vocab + 1 for model
    per_seq_batch_sample=config_from_file.get("per_seq_batch_sample", False),  # Flag for per-sequence batch sampling
)

# Update the main config with command-line arguments and W&B run ID if available
config["device"] = args.device
config["savename"] = args.savename
config["metadata_file"] = args.metadata_file

if wandb and wandb.run:
    config["wandb_run_id"] = wandb.run.id
    wandb.config.update(config, allow_val_change=True) # Log the final combined config to W&B

config_hash = make_hash(config)

checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
set_seed(config["seed"])


# dataset_name = config["pretraining_dataset_name"] # Removed
# save_dir = Path(f"save/dev_{config['savename']}-dataset_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")  # Use config savename
# Simplified save_dir as dataset_name (derived from data_type) is removed
save_dir = Path(f"save/dev_{config['savename']}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True) # Ensure save_dir is created before saving config/vocab

# Function to save config (if not in methylgpt.utils.logging)
def save_config(config_dict, path):
    with open(path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, default=str) # Use default=str for Path objects

save_config(config, save_dir)

logger = setup_logger("logger", save_dir / "run.log")
add_console_handler(logger)
train_logger = setup_logger("train_logger", save_dir / "train.log")
add_console_handler(train_logger)
test_logger = setup_logger("test_logger", save_dir / "test.log")
add_console_handler(test_logger)

# Ensure the vocab is saved to the correct final location
if not (save_dir / "methyl_vocab.pkl").exists():
    methyl_vocab_instance.save_dir = save_dir
    methyl_vocab_instance._save_vocab()
    logger.info(f"MethylVocab saved to {save_dir}")
else:
    methyl_vocab_instance.save_dir = save_dir # Ensure it's set even if loaded
    logger.info(f"MethylVocab already exists at {save_dir} or will be loaded/used from there.")


parquet_dirs = []
if config["parquet_dir"].exists() and config["parquet_dir"].is_dir():
    parquet_dirs = [
        os.path.join(config["parquet_dir"], f) for f in os.listdir(config["parquet_dir"]) if f.endswith(".parquet")
    ]
else:
    logger.error(f"Parquet directory {config['parquet_dir']} does not exist or is not a directory.")

logger.info(f"Number of parquet files found: {len(parquet_dirs)}")
if not parquet_dirs:
    logger.error("No parquet files found. Please check the preprocessing step and parquet_data_dir path.")
    # Depending on desired behavior, could exit here:
    # sys.exit("Exiting due to no parquet files found.")


train_files, valid_files = split_files(parquet_dirs, valid_ratio=config["valid_ratio"])
logger.info(f"Loading data from {len(train_files)} training files and {len(valid_files)} validation files")

train_dataloader = create_dataloader(
    train_files,
    config["batch_size"],
    # config["pad_value"], # Removed: pad_value is no longer an argument to create_dataloader
    num_workers=config_from_file.get("num_dataloader_workers", 0)
)
valid_dataloader = create_dataloader(
    valid_files,
    config["batch_size"],
    # config["pad_value"], # Removed: pad_value is no longer an argument to create_dataloader
    num_workers=config_from_file.get("num_dataloader_workers", 0)
)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
config["device"] = str(device) # Update config with actual device used

model = MethylGPTModel(
    config=config,  # Pass the entire config dictionary
    vocab=methyl_vocab_instance
)
model.to(device)

if config["pretrained_file"] is not None:
    try:
        model.load_state_dict(torch.load(config["pretrained_file"], map_location=device))
        logger.info(f"Loaded pretrained model from {config['pretrained_file']}")
    except FileNotFoundError:
        logger.error(f"Pretrained model file not found: {config['pretrained_file']}")
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")


if wandb and wandb.run:
    logger.info("W&B watching model (config logged).")
    wandb.config.update(config, allow_val_change=True)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["schedule_ratio"]) # step_size=1 if called every epoch

scaler = None
if config["amp"] and str(device).startswith("cuda"):
    if amp_available:
        scaler = GradScaler()
        logger.info("AMP GradScaler initialized.")
    else: # amp was configured True but torch.cuda.amp.GradScaler is not available
        logger.warning("AMP was configured but GradScaler is not available. Disabling AMP for training.")
        config["amp"] = False # Correctly disable AMP if scaler cannot be used.
else: # AMP not configured or device is CPU
    config["amp"] = False


# Comment out or remove old train/evaluate functions if they are no longer used
# and are causing lint errors (e.g. for 'criterion')
"""
def train(model, dataloader, optimizer, criterion, device, config, train_logger):
    # ... old train function ...
    pass

def evaluate(model, dataloader, optimizer, criterion, device, config, test_logger,  epoch=0):
    # ... old evaluate function ...
    pass
"""

# Load and process metadata (if needed for UMAP later, keep this part)
# def load_metadata_from_data_dir(metadata_file_path):
#     # ... (this function seems okay if compiled_data is used later) ...
# compiled_data = load_metadata_from_data_dir(config["metadata_file"])


# Checkpoint loading
latest_checkpoint_path = None # Define before try block
checkpoint_dir_path = Path(checkpoint_dir) # Ensure checkpoint_dir is a Path
if checkpoint_dir_path.exists():
    checkpoint_files = list(checkpoint_dir_path.glob(f'checkpoint_{config_hash}_*.pth'))
    if checkpoint_files:
        checkpoints_with_epochs = []
        for path in checkpoint_files:
            try:
                epoch_num = int(path.stem.split('_')[-1])
                checkpoints_with_epochs.append((path, epoch_num))
            except ValueError:
                logger.warning(f"Could not parse epoch from checkpoint filename: {path.name}")
        if checkpoints_with_epochs:
            latest_checkpoint_path, latest_epoch = max(checkpoints_with_epochs, key=lambda x: x[1])
            latest_checkpoint_path = str(latest_checkpoint_path) # Convert Path to string for torch.load

start_epoch = 1
if latest_checkpoint_path is not None:
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} from {latest_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error loading checkpoint {latest_checkpoint_path}: {e}. Starting from scratch.")
        start_epoch = 1
else:
    logger.info("No checkpoint found. Starting from scratch.")


# Main training loop
best_val_loss = float("inf")
best_model_state = None # Store state_dict instead of full model
best_model_epoch = 0

logger.info(f"Starting training from epoch {start_epoch} to {config['epochs']}")

for epoch in range(start_epoch, config["epochs"] + 1):
    epoch_start_time = time.time()
    logger.info(f"--- Epoch {epoch}/{config['epochs']} ---")

    # Training phase
    model.train()
    total_train_loss = 0
    total_train_mse = 0 # For tracking MSE component
    total_train_gepc = 0 # For tracking GEPC component (if used)
    processed_train_samples = 0 # Count samples for weighted average
    
    pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=False)
    for i, batch in enumerate(pbar_train):
        optimizer.zero_grad()

        # Prepare data using the model's method
        # batch["data"] is already on the correct device if dataloader sends it there,
        # but prepare_data might expect CPU data or handle device transfer internally.
        # For now, assume batch elements are on CPU, and prepare_data will return tensors.
        # The model.prepare_data method should handle moving data to the correct device if necessary,
        # or return tensors that can then be moved.
        
        # The batch from DataLoader contains {'id': [...], 'data': [np.array, np.array, ...]}
        # model.prepare_data expects the raw batch dictionary.
        prepared_batch = model.prepare_data(batch) 

        input_gene_ids = prepared_batch["gene_ids"].to(device)
        input_values = prepared_batch["values"].to(device) # These are masked
        target_values = prepared_batch["target_values"].to(device) # These are original, padded
        
        # Create src_key_padding_mask from target_values (or gene_ids if padding is consistent)
        # Pad token ID from vocab is used to identify padding in gene_ids
        pad_token_id = methyl_vocab_instance[config["pad_token"]]
        src_key_padding_mask = input_gene_ids.eq(pad_token_id) # Mask where gene_id is pad_token_id

        with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)):
            output_dict = model(
                input_gene_ids, 
                input_values, # Masked values
                src_key_padding_mask=src_key_padding_mask,
                MVC=config["GEPC"],       # Pass GEPC flag from config
                ECS=config["ecs_thres"] > 0 # Pass ECS flag from config
            )
            
            # Calculate loss based on the original pretraining script logic
            # masked_mse_loss expects (predictions, targets, mask_for_loss_calculation)
            # The mask for loss should identify positions that were originally masked *and* not padding.
            
            # loss_positions: True where loss should be calculated
            # True for positions that were masked (input_values == mask_value) AND are not padding tokens (target_values != pad_value)
            is_masked_for_input = input_values.eq(config["mask_value"]) 
            is_not_padding_target = target_values.ne(config["pad_value"])
            loss_positions = torch.logical_and(is_masked_for_input, is_not_padding_target)

            loss_mse = masked_mse_loss(output_dict["mlm_output"], target_values, loss_positions)
            loss = loss_mse

            if config["GEPC"] and "mvc_output" in output_dict and output_dict["mvc_output"] is not None:
                loss_gepc = masked_mse_loss(output_dict["mvc_output"], target_values, loss_positions)
                loss = loss + loss_gepc
            else:
                loss_gepc = torch.tensor(0.0).to(device) # So it can be added to total

        if scaler is not None:
            scaler.scale(loss).backward()
            # Consider gradient clipping here if it was in the original and scaler is used
            # scaler.unscale_(optimizer) # Optional if you need to inspect/clip grads before optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Example clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Example clipping
            optimizer.step()

        total_train_loss += loss.item() * input_gene_ids.size(0) # Weighted by batch size
        total_train_mse += loss_mse.item() * input_gene_ids.size(0)
        if config["GEPC"]:
            total_train_gepc += loss_gepc.item() * input_gene_ids.size(0)
        processed_train_samples += input_gene_ids.size(0)
        
        if i % config["log_interval"] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss_so_far = total_train_loss / processed_train_samples if processed_train_samples > 0 else 0
            pbar_train.set_postfix_str(f"Batch Loss: {loss.item():.4f}, Avg Loss: {avg_loss_so_far:.4f}, LR: {current_lr:.2e}")
            if wandb and wandb.run:
                log_dict = {
                    "train_loss_batch": loss.item(), 
                    "train_mse_batch": loss_mse.item(),
                    "learning_rate": current_lr,
                    "epoch_progress": epoch + i / len(train_dataloader)
                }
                if config["GEPC"]:
                    log_dict["train_gepc_batch"] = loss_gepc.item()
                wandb.log(log_dict, step=epoch * len(train_dataloader) + i) # Global step

    avg_train_loss = total_train_loss / processed_train_samples if processed_train_samples > 0 else 0
    avg_train_mse = total_train_mse / processed_train_samples if processed_train_samples > 0 else 0
    avg_train_gepc = total_train_gepc / processed_train_samples if processed_train_samples > 0 else 0
    
    train_logger.info(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | Avg Train MSE: {avg_train_mse:.4f} | Avg Train GEPC: {avg_train_gepc:.4f}")
    if wandb and wandb.run:
        log_dict_epoch = {
            "avg_train_loss_epoch": avg_train_loss,
            "avg_train_mse_epoch": avg_train_mse,
            "epoch": epoch
        }
        if config["GEPC"]:
            log_dict_epoch["avg_train_gepc_epoch"] = avg_train_gepc
        wandb.log(log_dict_epoch)

    # Validation phase
    model.eval()
    total_valid_loss = 0
    total_valid_mse = 0
    total_valid_gepc = 0
    processed_valid_samples = 0
    
    pbar_valid = tqdm(valid_dataloader, desc=f"Validating Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar_valid:
            prepared_batch_valid = model.prepare_data(batch)

            input_gene_ids_valid = prepared_batch_valid["gene_ids"].to(device)
            input_values_valid = prepared_batch_valid["values"].to(device)
            target_values_valid = prepared_batch_valid["target_values"].to(device)

            pad_token_id_valid = methyl_vocab_instance[config["pad_token"]]
            src_key_padding_mask_valid = input_gene_ids_valid.eq(pad_token_id_valid)
            
            with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)): 
                output_dict_valid = model(
                    input_gene_ids_valid, 
                    input_values_valid, 
                    src_key_padding_mask=src_key_padding_mask_valid,
                    MVC=config["GEPC"],
                    ECS=config["ecs_thres"] > 0
                )

                is_masked_for_input_valid = input_values_valid.eq(config["mask_value"])
                is_not_padding_target_valid = target_values_valid.ne(config["pad_value"])
                loss_positions_valid = torch.logical_and(is_masked_for_input_valid, is_not_padding_target_valid)

                loss_mse_valid = masked_mse_loss(output_dict_valid["mlm_output"], target_values_valid, loss_positions_valid)
                loss_valid = loss_mse_valid

                if config["GEPC"] and "mvc_output" in output_dict_valid and output_dict_valid["mvc_output"] is not None:
                    loss_gepc_valid = masked_mse_loss(output_dict_valid["mvc_output"], target_values_valid, loss_positions_valid)
                    loss_valid = loss_valid + loss_gepc_valid
                else:
                    loss_gepc_valid = torch.tensor(0.0).to(device)
            
            total_valid_loss += loss_valid.item() * input_gene_ids_valid.size(0)
            total_valid_mse += loss_mse_valid.item() * input_gene_ids_valid.size(0)
            if config["GEPC"]:
                total_valid_gepc += loss_gepc_valid.item() * input_gene_ids_valid.size(0)
            processed_valid_samples += input_gene_ids_valid.size(0)
    
    avg_valid_loss = total_valid_loss / processed_valid_samples if processed_valid_samples > 0 else 0
    avg_valid_mse = total_valid_mse / processed_valid_samples if processed_valid_samples > 0 else 0
    avg_valid_gepc = total_valid_gepc / processed_valid_samples if processed_valid_samples > 0 else 0

    test_logger.info(f"Epoch {epoch} | Avg Valid Loss: {avg_valid_loss:.4f} | Avg Valid MSE: {avg_valid_mse:.4f} | Avg Valid GEPC: {avg_valid_gepc:.4f}")
    if wandb and wandb.run:
        log_dict_epoch_val = {
            "avg_valid_loss_epoch": avg_valid_loss,
            "avg_valid_mse_epoch": avg_valid_mse,
            "epoch": epoch
        }
        if config["GEPC"]:
            log_dict_epoch_val["avg_valid_gepc_epoch"] = avg_valid_gepc
        wandb.log(log_dict_epoch_val)

    # Learning rate scheduling
    lr_scheduler.step()

    # Checkpointing
    # Ensure the specific checkpoint directory for this run exists
    current_run_checkpoint_dir = save_dir / "checkpoints"
    current_run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if epoch % config["save_eval_interval"] == 0 or epoch == config["epochs"]:
        # Save model checkpoint
        checkpoint_filename = f"checkpoint_{config_hash}_{epoch}.pth"
        checkpoint_path = current_run_checkpoint_dir / checkpoint_filename
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint at epoch {epoch}: {e}")

    # Early stopping or other criteria could be checked here
    # For now, we just log the best model based on validation loss
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_model_epoch = epoch
        logger.info(f"New best model found at epoch {epoch} with validation loss: {best_val_loss:.4f}")

# At the end of training, load the best model state if available
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    logger.info(f"Loaded best model from epoch {best_model_epoch} with validation loss: {best_val_loss:.4f}")
else:
    logger.warning("No best model found (best_model_state is None). Training may not have completed successfully.")

# Final evaluation on the validation set with the best model (or last model if no improvement)
# This section might need adjustment if UMAP or other specific evaluations are required as per original.
# For now, it mirrors the validation loop structure for loss calculation.
model.eval()
total_final_valid_loss = 0
total_final_valid_mse = 0
total_final_valid_gepc = 0
processed_final_valid_samples = 0

pbar_final_valid = tqdm(valid_dataloader, desc="Final Validation", leave=False)
with torch.no_grad():
    for batch in pbar_final_valid:
        prepared_batch_final = model.prepare_data(batch)

        input_gene_ids_final = prepared_batch_final["gene_ids"].to(device)
        input_values_final = prepared_batch_final["values"].to(device)
        target_values_final = prepared_batch_final["target_values"].to(device)

        pad_token_id_final = methyl_vocab_instance[config["pad_token"]]
        src_key_padding_mask_final = input_gene_ids_final.eq(pad_token_id_final)
        
        with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)):
            output_dict_final = model(
                input_gene_ids_final, 
                input_values_final, 
                src_key_padding_mask=src_key_padding_mask_final,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0
            )

            is_masked_for_input_final = input_values_final.eq(config["mask_value"])
            is_not_padding_target_final = target_values_final.ne(config["pad_value"])
            loss_positions_final = torch.logical_and(is_masked_for_input_final, is_not_padding_target_final)

            loss_mse_final = masked_mse_loss(output_dict_final["mlm_output"], target_values_final, loss_positions_final)
            loss_final = loss_mse_final

            if config["GEPC"] and "mvc_output" in output_dict_final and output_dict_final["mvc_output"] is not None:
                loss_gepc_final = masked_mse_loss(output_dict_final["mvc_output"], target_values_final, loss_positions_final)
                loss_final = loss_final + loss_gepc_final
            else:
                loss_gepc_final = torch.tensor(0.0).to(device)
        
        total_final_valid_loss += loss_final.item() * input_gene_ids_final.size(0)
        total_final_valid_mse += loss_mse_final.item() * input_gene_ids_final.size(0)
        if config["GEPC"]:
            total_final_valid_gepc += loss_gepc_final.item() * input_gene_ids_final.size(0)
        processed_final_valid_samples += input_gene_ids_final.size(0)

avg_final_valid_loss = total_final_valid_loss / processed_final_valid_samples if processed_final_valid_samples > 0 else 0
avg_final_valid_mse = total_final_valid_mse / processed_final_valid_samples if processed_final_valid_samples > 0 else 0
avg_final_valid_gepc = total_final_valid_gepc / processed_final_valid_samples if processed_final_valid_samples > 0 else 0

logger.info(f"Final Validation | Avg Loss: {avg_final_valid_loss:.4f} | Avg MSE: {avg_final_valid_mse:.4f} | Avg GEPC: {avg_final_valid_gepc:.4f}")

# If using W