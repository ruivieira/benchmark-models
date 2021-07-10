# -*- coding: utf-8 -*-
import click
import colorlog
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import joblib

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)
logger = colorlog.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Trains a model from processed data into a joblib-serialised model"""

    # load processed dataset
    logger.info("Loading processed dataset")
    _df = pd.read_csv(os.path.join(input_filepath, "train.csv"))

    X_df = _df.drop("PaidLoan", axis=1)
    y_df = _df["PaidLoan"]

    train_x, _, train_y, _ = train_test_split(
        X_df, y_df, test_size=0.25, random_state=42
    )

    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()

    param_test = {
        "max_depth": [1, 4, 8],
        "learning_rate": [0.05, 0.06, 0.07],
        "n_estimators": [10, 100, 200],
    }

    gsearch = GridSearchCV(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="error",
            scale_pos_weight=scale_pos_weight,
            seed=27,
            use_label_encoder=False,
        ),
        param_grid=param_test,
        scoring="roc_auc",
        n_jobs=-1,
        cv=8,
        verbose=10,
    )

    gsearch.fit(train_x, train_y)

    best_params, best_score = gsearch.best_params_, gsearch.best_score_
    logger.info("Best Parameters: {} | Best AUC: {}".format(best_params, best_score))

    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="error",
        scale_pos_weight=scale_pos_weight,
        seed=27,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        use_label_encoder=False,
    )

    xgb_model.fit(train_x, train_y)

    joblib.dump(xgb_model, output_filepath)
    logger.info("Saved model")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
