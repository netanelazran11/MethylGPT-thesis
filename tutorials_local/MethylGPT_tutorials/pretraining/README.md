# Tutorial: Generating and Visualizing Embeddings

This folder contains Jupyter notebooks for:
1. **Generating cell embeddings** using a pretrained methylGPT model.
2. **Visualizing** those embeddings in UMAP space.

## Files

- **`get_embeddings.ipynb`**  
  Generates embeddings (output of the `<bos>` token) from a pretrained methylGPT model.  
  - **Output**: Embeddings are saved in the `Embeddings/` folder.

- **`plot_embeddings.ipynb`**  
  Loads the generated embeddings and applies UMAP for dimensionality reduction.  
  - **Output**: UMAP plots are saved in the `Figures/` folder.

## Usage

1. **Generate Embeddings**  
   - Run `get_embeddings.ipynb` to produce the embedding files in `Embeddings/`.

2. **Plot Embeddings**  
   - Run `plot_embeddings.ipynb` to visualize the embeddings in UMAP space and save the figures in `Figures/`.

## Notes

- Created on **Jan 22** by **Jinyeop Song** (yeopjin@mit.edu).
- Ensure the `Embeddings/` and `Figures/` directories exist or update paths accordingly before running the notebooks.

## Pretraining the Model

The pretraining process involves data preprocessing and model training. A shell script, `run_pretraining.sh`, is provided in the `tutorials/pretraining/` directory to automate these steps.

### Running the Pretraining Script

**Setup and Execution:**

1.  **Navigate to the Pretraining Directory:**
    Open your terminal and change to the pretraining directory. Adjust `path/to/your/MethylGPT` to your actual project path:
    ```bash
    cd path/to/your/MethylGPT/tutorials/pretraining
    ```
    All subsequent commands assume you are in this `tutorials/pretraining/` directory.

2.  **Environment:**
    Ensure you have a suitable Python environment (e.g., using Conda) activated with all necessary dependencies. If a `requirements.txt` file is provided in this directory or the project root, install the dependencies from it.

3.  **Make Scripts Executable:**
    Ensure the necessary shell scripts are executable:
    ```bash
    chmod +x download_files.sh
    chmod +x run_pretraining.sh
    ```
    *(The `download_files.sh` script is for fetching example data).*

4.  **Download Example Data (Optional):**
    If you want to get started quickly with minimal example data, run the `download_files.sh` script:
    ```bash
    bash download_files.sh
    ```
    This script is expected to download sample data (e.g., into a `data_examples` subdirectory) and a sample probe ID file (e.g., `probe_ids_type3.csv`) into the current directory (`tutorials/pretraining/`).
    *After running, verify that the `RAW_DATA_DIR` and `PROBE_ID_REF` variables in `run_pretraining.sh` correctly point to the downloaded data's location and filenames.*

5.  **Prepare Your Data (If not using example data):**
    *   If you are using your own dataset, ensure your raw methylation data (typically CSV or CSV.gz files) is accessible. By default, `run_pretraining.sh` expects this data to be in a subdirectory named `data_examples` within the current (`tutorials/pretraining/`) directory.
    *   Ensure your probe ID reference file (e.g., `probe_ids_type3.csv` by default) is also present in the current (`tutorials/pretraining/`) directory.
    *   If your data is located elsewhere or uses different filenames, you **must** update the `RAW_DATA_DIR` and `PROBE_ID_REF` variables at the top of the `run_pretraining.sh` script.

6.  **Configuration File:**
    A training configuration JSON file (default: `config_example.json`) is expected in the current (`tutorials/pretraining/`) directory. Review and modify this file to suit your experiment's requirements (e.g., model architecture, learning rate).

7.  **Review `run_pretraining.sh` Script:**
    Open `run_pretraining.sh` in a text editor. Carefully review and adjust the user-configurable variables at the top, especially those marked with a `# TOCHANGE` comment:
    *   `RUN_PREPROCESSING`: Set to `true` to execute the preprocessing step. Set to `false` if your data is already preprocessed and in the expected format/location.
    *   `RAW_DATA_DIR`: Path to the directory containing your raw input data.
    *   `PROBE_ID_REF`: Path to your probe ID reference file.
    *   `CONFIG_FILE`: Name of the training configuration JSON file.
    *   `PREPROCESS_SCRIPT`: Name of the Python script for preprocessing.
    *   `TRAIN_SCRIPT`: Name of the Python script for training.
    *   `DEVICE`: Specify the CUDA device for training (e.g., `cuda:0`).
    *   `SAVENAME_PREFIX`: A prefix used for naming the output directory where model checkpoints and logs will be saved. A timestamp is automatically appended to this prefix.

8.  **Run the Pretraining Script:**
    Once everything is configured, execute the main pretraining script:
    ```bash
    ./run_pretraining.sh
    ```

**Output:**

*   **Preprocessing:** If `RUN_PREPROCESSING` is `true`, the script will generate:
    *   A metadata file (e.g., `QCed_samples_type3.csv` by default) in the current directory.
    *   Preprocessed data in Parquet format, saved into a subdirectory (e.g., `parquet_files/` by default) within the current directory.
*   **Training:**
    *   Model checkpoints, logs, and any other training artifacts will be saved in a new directory. The name of this directory will be constructed from `SAVENAME_PREFIX` followed by a timestamp (e.g., `pretrain_run_YYYYMMDD_HHMMSS/`).
*   The script will print its configuration settings and the commands it executes to the console, providing a log of its operations.

**Adjusting Training Parameters:**

For modifying training parameters such as learning rate, batch size, model architecture details, etc., it is recommended to directly update the values within the JSON configuration file (e.g., `config_example.json`). This file provides a centralized place for all training-specific settings.

Make sure to review the contents of your chosen configuration file (specified by the `CONFIG_FILE` variable in `run_pretraining.sh`) and adjust the parameters as needed for your experiment before running the training script.
