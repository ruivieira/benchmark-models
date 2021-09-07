import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import click
from nyoka.skl.skl_to_pmml import skl_to_pmml
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np
from sklearn2pmml import sklearn2pmml


def build_RF_pipeline(inputs, outputs, categorical, numerical, rf=None):
    if not rf:
        rf = RandomForestClassifier()
    pipeline = Pipeline(
        [
            (
                "mapper",
                DataFrameMapper(
                    [(categorical, preprocessing.OrdinalEncoder()), (numerical, None)]
                ),
            ),
            ("classifier", rf),
        ]
    )
    pipeline.fit(inputs, outputs)
    return pipeline


def RF_estimation(
    inputs,
    outputs,
    estimator_steps=10,
    depth_steps=10,
    min_samples_split=None,
    min_samples_leaf=None,
):
    # hyper-parameter estimation
    n_estimators = [
        int(x) for x in np.linspace(start=50, stop=100, num=estimator_steps)
    ]
    max_depth = [int(x) for x in np.linspace(3, 10, num=depth_steps)]
    max_depth.append(None)
    if not min_samples_split:
        min_samples_split = [2, 3, 4]
    if not min_samples_leaf:
        min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(inputs, outputs)
    best_random = rf_random.best_estimator_
    print(best_random)
    return best_random


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
    model = RandomForestClassifier(verbose=True, max_depth=6, n_jobs=-1)
    logger.info("Fitting model")
    model.fit(X_train, y_train)
    logger.info("Saving model")
    dump(model, model_dest)

    pipeline = PMMLPipeline(
        [("classifier", model)]
    )

    logger.info("Saving PMML model")
    skl_to_pmml(
        pipeline,
        [
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AGE",
            "DAYS_EMPLOYED",
            "FLAG_WORK_PHONE",
        ],
        "APPROVED",
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