#!/usr/bin/env python3
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import gzip
import os
import numpy as np

CELL_CHUNK_SIZE = 5000

def process_to_cell_chunked_parquet(input_raw_csv_dir_str: str, probe_id_ref_path_str: str, output_metadata_dir_str: str, output_parquet_base_dir_str: str):
    input_raw_csv_dir = Path(input_raw_csv_dir_str)
    probe_id_ref_path = Path(probe_id_ref_path_str)
    output_metadata_dir = Path(output_metadata_dir_str)
    output_parquet_dir = Path(output_parquet_base_dir_str) / "parquet_files"

    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    output_parquet_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input raw CSV directory: {input_raw_csv_dir}")
    print(f"Reference probe ID CSV: {probe_id_ref_path}")
    print(f"Output metadata directory: {output_metadata_dir}")
    print(f"Output Parquet directory (for cell chunks): {output_parquet_dir}")

    try:
        probe_ids_df = pd.read_csv(probe_id_ref_path)
        if "illumina_probe_id" not in probe_ids_df.columns:
            raise ValueError("Reference probe ID CSV must contain 'illumina_probe_id' column.")
        master_probe_id_order = probe_ids_df["illumina_probe_id"].astype(str).tolist()
        master_probe_id_set = set(master_probe_id_order)
        print(f"Loaded {len(master_probe_id_order)} reference probe IDs.")
    except Exception as e:
        print(f"Error loading reference probe IDs: {e}")
        return

    all_cells_data_list = []

    # Collect all .csv and .csv.gz files
    raw_csv_files = []
    for f in sorted(input_raw_csv_dir.iterdir()):
        if f.is_file():
            if f.suffix == ".csv" or f.suffixes == [".csv", ".gz"]:
                raw_csv_files.append(f)
            else:
                print(f"Skipping non-CSV file: {f.name}")

    print(f"Found {len(raw_csv_files)} raw CSV files to process.")

    for csv_file_path in raw_csv_files:
        print(f"Processing file: {csv_file_path.name}")
        try:
            if csv_file_path.name.endswith(".gz"):
                with gzip.open(csv_file_path, 'rt') as f:
                    df_raw = pd.read_csv(f, low_memory=False)
            else:
                df_raw = pd.read_csv(csv_file_path, low_memory=False)

            if df_raw.empty:
                print(f"  Skipping empty file: {csv_file_path.name}")
                continue

            probe_col = df_raw.columns[0]
            df_raw.rename(columns={probe_col: 'probe_id'}, inplace=True)
            df_raw['probe_id'] = df_raw['probe_id'].astype(str)
            df_raw.set_index('probe_id', inplace=True)
            sample_columns = df_raw.columns

            for cell_id in sample_columns:
                ordered_values = [
                    pd.to_numeric(df_raw.loc[probe_id, cell_id], errors='coerce') if probe_id in df_raw.index else np.nan
                    for probe_id in master_probe_id_order
                ]
                all_cells_data_list.append({"id": str(cell_id), "data": ordered_values})

            print(f"  Finished extracting {len(sample_columns)} samples.")
        except Exception as e:
            print(f"  Error processing file {csv_file_path.name}: {e}")
            continue

    if not all_cells_data_list:
        print("No valid data found.")
        return

    all_cells_df = pd.DataFrame(all_cells_data_list)
    print(f"Total cells processed: {len(all_cells_df)}")

    final_parquet_chunk_metadata = []
    for i in range(0, len(all_cells_df), CELL_CHUNK_SIZE):
        chunk_df = all_cells_df.iloc[i:i + CELL_CHUNK_SIZE]
        chunk_num = i // CELL_CHUNK_SIZE
        parquet_chunk_filename = f"all_cells_chunk_{chunk_num}.parquet"
        parquet_path = output_parquet_dir / parquet_chunk_filename

        try:
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            pq.write_table(table, parquet_path)
            print(f"  Saved: {parquet_path.name} with {len(chunk_df)} cells.")
            final_parquet_chunk_metadata.append({
                'chunk_file_name': parquet_chunk_filename,
                'path': str(parquet_path.resolve()),
                'num_cells_in_chunk': len(chunk_df),
                'num_features_per_cell': len(master_probe_id_order)
            })
        except Exception as e:
            print(f"    Error writing chunk {parquet_chunk_filename}: {e}")
            continue

    if final_parquet_chunk_metadata:
        metadata_df = pd.DataFrame(final_parquet_chunk_metadata)
        qced_output_path = output_metadata_dir / "QCed_samples_type3.csv"
        metadata_df.to_csv(qced_output_path, index=False)
        print(f"Metadata written to: {qced_output_path}")
    else:
        print("No chunks saved. Skipping metadata.")

    print("✅ Done: All valid CSV files processed and saved to Parquet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw methylation CSV data to chunked Parquet files.")
    parser.add_argument("--input_raw_csv_dir", type=str, required=True, help="Directory with raw CSV or CSV.gz files.")
    parser.add_argument("--probe_id_ref_path", type=str, required=True, help="Reference CSV with 'illumina_probe_id' column.")
    parser.add_argument("--output_metadata_dir", type=str, required=True, help="Directory for output metadata CSV.")
    parser.add_argument("--output_parquet_base_dir", type=str, required=True, help="Base directory for parquet_files/")

    args = parser.parse_args()

    process_to_cell_chunked_parquet(
        args.input_raw_csv_dir,
        args.probe_id_ref_path,
        args.output_metadata_dir,
        args.output_parquet_base_dir
    )
