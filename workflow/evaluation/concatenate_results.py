"""Concatenate evaluation result files into a single data table."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_files, to_numeric  # noqa: E402

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
    to_int = snakemake.params["to_int"]
    to_float = snakemake.params["to_float"]
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
    to_int = args.to_int
    to_float = args.to_float

#%% Load
data_table = load_files(input_files).fillna("")

# %% Type conversion
data_table = to_numeric(data_table, to_int, to_float)
data_table = data_table.rename(columns={"K": "q"})

# %% Save
data_table.to_csv(output_file)
