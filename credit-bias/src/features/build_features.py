# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import LabelBinarizer


def filter_training_columns(df_):
    training_columns = ['NewCreditCustomer', 'Amount',
                        'Interest', 'LoanDuration', 'Education',
                        'NrOfDependants', 'EmploymentDurationCurrentEmployer',
                        'IncomeFromPrincipalEmployer', 'IncomeFromPension',
                        'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'ExistingLiabilities', 'RefinanceLiabilities',
                        'DebtToIncome', 'FreeCash',
                        'CreditScoreEeMini', 'NoOfPreviousLoansBeforeLoan',
                        'AmountOfPreviousLoansBeforeLoan', 'PreviousRepaymentsBeforeLoan',
                        'PreviousEarlyRepaymentsBefoleLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan', 'PaidLoan',
                        'Council_house', 'Homeless', 'Joint_ownership', 'Joint_tenant',
                        'Living_with_parents', 'Mortgage', 'Other', 'Owner',
                        'Owner_with_encumbrance', 'Tenant', 'Entrepreneur',
                        'Fully', 'Partially', 'Retiree', 'Self_employed']

    return df_[training_columns]


def replace_columns(df_):
    new_values = {
        "NewCreditCustomer": {'Existing_credit_customer': 1, 'New_credit_Customer': 0},
        "Education": {'Higher': 5, 'Secondary': 4, 'Basic': 2, 'Vocational': 3, 'Primary': 1},
        "EmploymentDurationCurrentEmployer": {'MoreThan5Years': 6, 'UpTo3Years': 3, 'UpTo1Year': 1, 'UpTo5Years': 5,
                                              'UpTo2Years': 2, 'TrialPeriod': 0, 'UpTo4Years': 4, 'Retiree': 7,
                                              'Other': 0},
        "HomeOwnershipType": {'Tenant_unfurnished_property': 'Tenant', 'Tenant_pre_furnished_property': 'Tenant'}
    }
    return df_.replace(new_values)


def filter_rows(df_):
    df_ = df_[df_["EmploymentDurationCurrentEmployer"] != 0]
    return df_


def transform_columns_into_binary(df_, columns: list):
    for col in columns:
        lb_style = LabelBinarizer()
        lb_results = lb_style.fit_transform(df_[col].astype(str))
        binary_ = pd.DataFrame(lb_results, columns=lb_style.classes_)
        df_ = pd.concat([df_, binary_], axis=1, join='inner')
    return df_


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making training data set from interim data")

    # load iterim dataset
    _df = pd.read_csv(os.path.join(input_filepath, "data.csv"))
    _df = replace_columns(_df)
    _df = filter_rows(_df)
    _df = transform_columns_into_binary(_df, ['HomeOwnershipType', 'EmploymentStatus'])
    _df = filter_training_columns(_df)

    _df.to_csv(os.path.join(output_filepath, "train.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
