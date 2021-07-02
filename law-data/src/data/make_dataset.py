# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(os.path.join(input_filepath, "law-data.zip"))
    df = pd.get_dummies(df, columns=["race"], prefix="", prefix_sep="")
    df["male"] = df["sex"].map(lambda x: 1 if x == 2 else 0)
    df["female"] = df["sex"].map(lambda x: 1 if x == 1 else 0)
    df = df.drop(axis=1, columns=["sex"])
    df["LSAT"] = df["LSAT"].astype(int)
    df.to_csv(os.path.join(output_filepath, "data.csv"), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
