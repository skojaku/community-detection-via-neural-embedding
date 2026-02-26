"""Concatenate evaluation result files into a single data table."""
import argparse
import os
import sys

# Allow imports from the parent workflow package (e.g., workflow/utils).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_files, to_numeric  # noqa: E402

# --- Parse inputs from Snakemake or command-line arguments ---
if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
    int_columns = snakemake.params["to_int"]
    float_columns = snakemake.params["to_float"]
else:
    parser = argparse.ArgumentParser(
        description="Concatenate evaluation result files into a single CSV."
    )
    parser.add_argument("input_files", nargs="+", help="Input result files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--to-int", nargs="*", default=[], help="Columns to cast to int"
    )
    parser.add_argument(
        "--to-float", nargs="*", default=[], help="Columns to cast to float"
    )
    args = parser.parse_args()
    input_files = args.input_files
    output_file = args.output
    int_columns = args.to_int
    float_columns = args.to_float

# --- Load, convert types, and save ---
data_table = load_files(input_files).fillna("")
data_table = to_numeric(data_table, int_columns, float_columns)

# Rename "K" (number of communities parameter) to "q" for consistency.
data_table = data_table.rename(columns={"K": "q"})

data_table.to_csv(output_file)
