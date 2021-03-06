# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_file = os.path.join(input_filepath, "data.zip")
    logger.info(f"importing zipped data {input_file}")
    df = pd.read_csv(input_file)
    df = df[
        ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "PTRATIO", "MEDV"]
    ]

    df = df.dropna()
    inputs = df[["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "PTRATIO"]]

    outputs = df[["MEDV"]]

    inputs_dest_file = os.path.join(output_filepath, "inputs.csv")
    outputs_dest_file = os.path.join(output_filepath, "outputs.csv")

    logger.info("exporting inputs.csv and outputs.csv")
    inputs.to_csv(inputs_dest_file, index=False)
    outputs.to_csv(outputs_dest_file, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
