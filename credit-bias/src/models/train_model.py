# -*- coding: utf-8 -*-
from typing import List
import click
from pandas.core.frame import DataFrame
import colorlog
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from xgboost.sklearn import XGBClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from nyoka.skl.skl_to_pmml import skl_to_pmml
import joblib
import numpy as np
from typing import List, Optional

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)
logger = colorlog.getLogger(__name__)


@click.command()
@click.option("-i", "--input_filepath", type=click.Path(exists=True))
@click.option("-o", "--output_filepath", type=click.Path())
@click.option("-f", "--output_format")
def main(input_filepath, output_filepath, output_format):
    print("Options")
    print(input_filepath)
    print(output_format)
    print(output_filepath)
    """Trains a model from processed data into a joblib-serialised model"""

    # load processed dataset
    logger.info("Loading processed dataset")
    _df: DataFrame = pd.read_csv(os.path.join(input_filepath, "train.csv"))

    if not _df.empty:

        X_df: Optional[DataFrame] = _df.drop("PaidLoan", axis=1)
        y_df: DataFrame = _df["PaidLoan"]

        train_x: np.ndarray
        train_y: np.ndarray
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
        logger.info(
            "Best Parameters: {} | Best AUC: {}".format(best_params, best_score)
        )

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

        if output_format == "json":
            xgb_model.save_model(output_filepath + ".json")
            logger.info("Saving model as JSON")
        elif output_format == "ubj":
            logger.info("Saving model as Universal Binary JSON (UBJ)")
            xgb_model.save_model(output_filepath + ".ubj")
        elif output_format == "pmml":
            logger.info("Saving model as PMML")
            pipeline = PMMLPipeline([("classifier", xgb_model)])
            logger.info("Saving PMML model")
            skl_to_pmml(
                pipeline,
                [
                    "NewCreditCustomer",
                    "Amount",
                    "Interest",
                    "LoanDuration",
                    "Education",
                    "NrOfDependants",
                    "EmploymentDurationCurrentEmployer",
                    "IncomeFromPrincipalEmployer",
                    "IncomeFromPension",
                    "IncomeFromFamilyAllowance",
                    "IncomeFromSocialWelfare",
                    "IncomeFromLeavePay",
                    "IncomeFromChildSupport",
                    "IncomeOther",
                    "ExistingLiabilities",
                    "RefinanceLiabilities",
                    "DebtToIncome",
                    "FreeCash",
                    "CreditScoreEeMini",
                    "NoOfPreviousLoansBeforeLoan",
                    "AmountOfPreviousLoansBeforeLoan",
                    "PreviousRepaymentsBeforeLoan",
                    "PreviousEarlyRepaymentsBefoleLoan",
                    "PreviousEarlyRepaymentsCountBeforeLoan",
                    "Council_house",
                    "Homeless",
                    "Joint_ownership",
                    "Joint_tenant",
                    "Living_with_parents",
                    "Mortgage",
                    "Other",
                    "Owner",
                    "Owner_with_encumbrance",
                    "Tenant",
                    "Entrepreneur",
                    "Fully",
                    "Partially",
                    "Retiree",
                    "Self_employed",
                ],
                "PaidLoan",
                output_filepath + ".pmml",
            )
        else:
            raise ValueError(f"Format {output_format} not supported.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
