# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import pandas as pd
from random import random


def calculate_approval(row):
    p = 1.0

    if row["AGE"] < 25:
        p = p - 0.2
    elif row["AGE"] >= 25 and row["AGE"] < 50:
        p = p - 0.1

    if row["CNT_CHILDREN"] == 0:
        p = p - 0.05
    elif row["CNT_CHILDREN"] > 0 and row["CNT_CHILDREN"] < 2:
        p = p - 0.1
    else:
        p = p - 0.15

    if row["AMT_INCOME_TOTAL"] < 100000:
        p = p - 0.4
    elif row["AMT_INCOME_TOTAL"] >= 100000 and row["AMT_INCOME_TOTAL"] < 200000:
        p = p - 0.2
    elif row["AMT_INCOME_TOTAL"] >= 200000 and row["AMT_INCOME_TOTAL"] < 300000:
        p = p - 0.1

    if row["DAYS_EMPLOYED"] < 365:
        p = p - 0.2
    elif row["DAYS_EMPLOYED"] >= 365 and row["DAYS_EMPLOYED"] < 2000:
        p = p - 0.1

    if row["FLAG_OWN_REALTY"] == 0.0:
        p = p - 0.06

    if row["FLAG_OWN_CAR"] == 0.0:
        p = p - 0.025

    return random() < p


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    input_file = os.path.join(input_filepath, "application_record.zip")
    logger.info(f"importing zipped data {input_file}")
    df = pd.read_csv(input_file)
    df["AGE"] = -df["DAYS_BIRTH"] / 365.0
    df["DAYS_EMPLOYED"] = -df["DAYS_EMPLOYED"]
    df = df.dropna()
    df["APPROVED"] = df.apply(calculate_approval, axis=1)
    # filter out extreme income values
    df = df[df["AMT_INCOME_TOTAL"] < 1e6]
    df.loc[:, ("APPROVED")] = df.loc[:, ("APPROVED")].eq(True).mul(1)
    df.loc[:, ("FLAG_OWN_CAR")] = df.loc[:, ("FLAG_OWN_CAR")].eq("Y").mul(1)
    df.loc[:, ("FLAG_OWN_REALTY")] = df.loc[:, ("FLAG_OWN_REALTY")].eq("Y").mul(1)

    inputs = df[
        [
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AGE",
            "DAYS_EMPLOYED",
            "FLAG_WORK_PHONE",
        ]
    ]

    outputs = df[["APPROVED"]]

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
