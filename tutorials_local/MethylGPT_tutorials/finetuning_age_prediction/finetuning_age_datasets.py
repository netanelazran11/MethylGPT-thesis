import os
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
import torch

from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

# Sanity checks: this dataset depends only on scGPT's tokenizer utilities.
# If these fail, scGPT is not properly available in the current environment.
assert callable(tokenize_and_pad_batch), "scGPT tokenizer `tokenize_and_pad_batch` is not available"
assert callable(random_mask_value), "scGPT masking `random_mask_value` is not available"


current_directory = Path(__file__).parent.absolute()


# ------------------------------------------------------------
# Robust path resolution helpers
# ------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a repo marker is found."""
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.cfg").exists():
            return p
    return start.parent


REPO_ROOT = find_repo_root(Path(__file__))

# Optional: base directory for fine-tuning parquet files (can be set in the environment)
# Example (Moriah):
#   export METHYLGPT_FINETUNE_DATA_ROOT=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data
FINETUNE_DATA_ROOT = Path(
    os.environ.get(
        "METHYLGPT_FINETUNE_DATA_ROOT",
        "/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data",
    )
)


def resolve_path(p: Union[str, Path], *bases: Path) -> Path:
    """Resolve a path that may be absolute or relative.

    If relative, try the provided bases (in order) and return the first existing path.
    If none exist, return the first base / p (best-effort).
    """
    p = Path(p)
    if p.is_absolute():
        return p
    for b in bases:
        cand = (b / p).resolve()
        if cand.exists():
            return cand
    # best-effort fallback
    return (bases[0] / p).resolve() if bases else p.resolve()


class SimpleVocab:
    """Minimal Vocab replacement for torchtext's Vocab.

    We only rely on:
      - __getitem__ for token -> index
      - set_default_index
      - attribute `stoi` (token->index)

    This avoids torchtext private C++ bindings (`torchtext._torchtext`) which are often
    incompatible across versions / builds on HPC.
    """

    def __init__(self, tokens: Sequence[str]):
        self.itos: List[str] = list(tokens)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self._default_index: int = 0

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self._default_index)

    def set_default_index(self, idx: int) -> None:
        self._default_index = int(idx)


class CollatableVocab(object):
    def __init__(self, model_args):
        self.model_args = model_args
        self.max_seq_len = model_args["n_hvg"] + 1
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        self.mask_value = -1
        self.pad_value = -2
        self.mask_ratio = model_args["mask_ratio"]
        self.mask_seed = model_args["mask_seed"]
        self.vocab, self.CpG_ids = self.set_vocab()

    def set_vocab(self):
        # `probe_id_dir` can be absolute or relative. We try a few common bases:
        # 1) REPO_ROOT (this repo)
        # 2) REPO_ROOT/methylGPT (original upstream layout)
        # 3) current_directory (tutorial folder)
        probe_rel = self.model_args["probe_id_dir"]
        probe_csv = resolve_path(probe_rel, REPO_ROOT, REPO_ROOT / "methylGPT", current_directory)

        CpG_list = pd.read_csv(probe_csv)["illumina_probe_id"].values.tolist()
        CpG_ids = len(self.special_tokens) + np.arange(len(CpG_list))
        vocab = SimpleVocab(self.special_tokens + CpG_list)
        vocab.set_default_index(vocab["<pad>"])
        return vocab, CpG_ids


class Age_Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab: CollatableVocab, df, scaler):
        self.vocab = vocab
        self.scaler = scaler
        self.gene_datas = df["data"].to_list()
        self.ages_label_norm = self.label_norm(df["age"].to_numpy())
        self.ages_label = df["age"].to_numpy()

    def __getitem__(self, index: int):
        gen_data = self.gene_datas[index]
        ages_label = torch.tensor(self.ages_label[index], dtype=torch.float32)
        ages_label_norm = self.ages_label_norm[index]
        return gen_data, ages_label, ages_label_norm

    def collater(self, batch):
        gen_datas, ages_labels, ages_label_norms = tuple(zip(*batch))
        gene_ids, masked_values, target_values = self.tokenize(torch.as_tensor(gen_datas))
        ages_labels = torch.stack(ages_labels)
        ages_label_norms = torch.stack(ages_label_norms)
        return gene_ids, masked_values, target_values, ages_labels, ages_label_norms

    def __len__(self):
        return len(self.ages_label)

    def tokenize(self, data: torch.Tensor):
        methyl_data = torch.nan_to_num(data, nan=self.vocab.pad_value)
        # scGPT tokenizer expects numpy
        methyl_np = methyl_data.detach().cpu().numpy()

        tokenized_data = tokenize_and_pad_batch(
            methyl_np,
            self.vocab.CpG_ids,
            max_len=self.vocab.max_seq_len,
            vocab=self.vocab.vocab,
            pad_token=self.vocab.pad_token,
            pad_value=self.vocab.pad_value,
            append_cls=True,
            include_zero_gene=True,
        )

        masked_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=self.vocab.mask_ratio,
            mask_value=self.vocab.mask_value,
            pad_value=self.vocab.pad_value,
            mask_seed=self.vocab.mask_seed,
        )

        return tokenized_data["genes"], masked_values, tokenized_data["values"]

    def label_norm(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(self.scaler.transform(data.reshape(-1, 1)), dtype=torch.float32)