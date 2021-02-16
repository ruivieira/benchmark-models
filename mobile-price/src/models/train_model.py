import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from nyoka import skl_to_pmml


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_data", type=click.Path())
@click.argument("model_dest", type=click.Path())
def main(input_data, output_data, model_dest):
    logger = logging.getLogger(__name__)
    logger.info("Loading input and output data")
    inputs = pd.read_csv(input_data)
    outputs = pd.read_csv(output_data)
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=0.4, random_state=23
    )

    model = RandomForestClassifier(verbose=True, n_jobs=-1)

    logger.info("Fitting model")
    model = model.fit(X_train, y_train["price_range"].ravel())

    logger.info("Saving joblib model")
    dump(model, model_dest + ".joblib")

    pipeline = PMMLPipeline(
        [("classifier", RandomForestClassifier(verbose=True, n_jobs=-1))]
    )
    pipeline.fit(X_train, y_train["price_range"].ravel())
    pipeline.verify(X_test.sample(n=10))

    logger.info("Saving PMML model")
    skl_to_pmml(
        pipeline,
        [
            "battery_power",
            "clock_speed",
            "fc",
            "int_memory",
            "m_dep",
            "mobile_wt",
            "n_cores",
            "pc",
            "px_height",
            "px_width",
            "ram",
            "sc_h",
            "sc_w",
            "talk_time",
        ],
        "price_range",
        model_dest + ".pmml",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()