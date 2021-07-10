# -*- coding: utf-8 -*-
import click
import colorlog
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)
logger = colorlog.getLogger(__name__)


def filter_columns(df_: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "LoanNumber",
        "ListedOnUTC",
        "UserName",
        "NewCreditCustomer",
        "LoanDate",
        "MaturityDate_Original",
        "MaturityDate_Last",
        "Age",
        "DateOfBirth",
        "Gender",
        "Country",
        "AppliedAmount",
        "Amount",
        "Interest",
        "LoanDuration",
        "MonthlyPayment",
        "UseOfLoan",
        "Education",
        "MaritalStatus",
        "NrOfDependants",
        "EmploymentStatus",
        "EmploymentDurationCurrentEmployer",
        "WorkExperience",
        "OccupationArea",
        "HomeOwnershipType",
        "IncomeFromPrincipalEmployer",
        "IncomeFromPension",
        "IncomeFromFamilyAllowance",
        "IncomeFromSocialWelfare",
        "IncomeFromLeavePay",
        "IncomeFromChildSupport",
        "IncomeOther",
        "IncomeTotal",
        "ExistingLiabilities",
        "RefinanceLiabilities",
        "DebtToIncome",
        "FreeCash",
        "DefaultDate",
        "Status",
        "CreditScoreEeMini",
        "NoOfPreviousLoansBeforeLoan",
        "AmountOfPreviousLoansBeforeLoan",
        "PreviousRepaymentsBeforeLoan",
        "PreviousEarlyRepaymentsBefoleLoan",
        "PreviousEarlyRepaymentsCountBeforeLoan",
    ]

    return df_[selected_columns]


def filter_rows(df_: pd.DataFrame) -> pd.DataFrame:
    return df_[(df_["Country"] == "EE") & (df_["Status"] != "Current")]


def rename_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.replace(-1, np.nan)

    zero_replacements = [
        "Age",
        "Education",
        "MaritalStatus",
        "EmploymentStatus",
        "OccupationArea",
        "CreditScoreEeMini",
    ]
    df_[zero_replacements] = df_[zero_replacements].replace(0.0, np.nan)

    value_replacements = {
        "UseOfLoan": {
            0: "Loan_consolidation",
            1: "Real_estate",
            2: "Home_improvement",
            3: "Business",
            4: "Education",
            5: "Travel",
            6: "Vehicle",
            7: "Other",
            8: "Health",
            101: "Working_capital_financing",
            102: "Purchase_of_machinery_equipment",
            103: "Renovation_of_real_estate",
            104: "Accounts_receivable_financing ",
            105: "Acquisition_of_means_of_transport",
            106: "Construction_finance",
            107: "Acquisition_of_stocks",
            108: "Acquisition_of_real_estate",
            109: "Guaranteeing_obligation ",
            110: "Other_business",
        },
        "Education": {
            1: "Primary",
            2: "Basic",
            3: "Vocational",
            4: "Secondary",
            5: "Higher",
        },
        "MaritalStatus": {
            1: "Married",
            2: "Cohabitant",
            3: "Single",
            4: "Divorced",
            5: "Widow",
        },
        "EmploymentStatus": {
            1: "Unemployed",
            2: "Partially",
            3: "Fully",
            4: "Self_employed",
            5: "Entrepreneur",
            6: "Retiree",
        },
        "NewCreditCustomer": {0: "Existing_credit_customer", 1: "New_credit_Customer"},
        "OccupationArea": {
            1: "Other",
            2: "Mining",
            3: "Processing",
            4: "Energy",
            5: "Utilities",
            6: "Construction",
            7: "Retail_and_wholesale",
            8: "Transport_and_warehousing",
            9: "Hospitality_and_catering",
            10: "Info_and_telecom",
            11: "Finance_and_insurance",
            12: "Real_estate",
            13: "Research",
            14: "Administrative",
            15: "Civil_service_and_military",
            16: "Education",
            17: "Healthcare_and_social_help",
            18: "Art_and_entertainment",
            19: "Agriculture_forestry_and_fishing",
        },
        "HomeOwnershipType": {
            0: "Homeless",
            1: "Owner",
            2: "Living_with_parents",
            3: "Tenant_pre_furnished_property",
            4: "Tenant_unfurnished_property",
            5: "Council_house",
            6: "Joint_tenant",
            7: "Joint_ownership",
            8: "Mortgage",
            9: "Owner_with_encumbrance",
            10: "Other",
        },
        "NrOfDependants": {"10Plus": 11},
        "Gender": {0: "Male", 1: "Female", 2: "Unknown"},
    }
    df_ = df_.replace(value_replacements)

    return df_


def add_new_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_["Defaulted"] = df_["DefaultDate"].apply(lambda x: 0 if pd.isnull(x) else 1)
    df_["PaidLoan"] = df_["Defaulted"].replace({0: 1, 1: 0})
    df_["LoanStatus"] = df_["PaidLoan"].apply(
        lambda x: "Paid back" if x == 1 else "Defaulted"
    )
    df_["AgeGroup"] = df_["Age"].apply(lambda x: "Under 40" if x < 40 else "Over 40")

    return df_


def reformat_columns(df_: pd.DataFrame) -> pd.DataFrame:
    df_["ListedOnUTC"] = pd.to_datetime(df_["ListedOnUTC"])
    df_["LoanDate"] = pd.to_datetime(df_["LoanDate"])
    df_["MaturityDate_Original"] = pd.to_datetime(df_["MaturityDate_Original"])
    df_["MaturityDate_Last"] = pd.to_datetime(df_["MaturityDate_Last"])
    df_["DateOfBirth"] = pd.to_datetime(df_["DateOfBirth"])
    df_["DefaultDate"] = pd.to_datetime(df_["DefaultDate"])

    df_["LoanDuration"] = pd.to_numeric(df_["LoanDuration"])
    df_["NrOfDependants"] = pd.to_numeric(df_["NrOfDependants"])

    df_["CreditScoreEeMini"] = df_["CreditScoreEeMini"].astype(str)
    df_["Defaulted"] = df_["Defaulted"].astype(bool)
    df_["PaidLoan"] = df_["PaidLoan"].astype(bool)

    return df_


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("making interim data set from raw data")

    # load raw dataset
    _df = pd.read_csv(os.path.join(input_filepath, "LoanData.zip"))
    # Filter not used columns
    _df = filter_columns(_df)
    # Filter not used rows
    _df = filter_rows(_df)
    # Rename columns
    _df = rename_columns(_df)
    # Add new columns
    _df = add_new_columns(_df)
    # Reformat columns
    _df = reformat_columns(_df)

    _df.to_csv(os.path.join(output_filepath, "data.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
