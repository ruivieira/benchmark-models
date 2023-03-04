# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_file = os.path.join(input_filepath, "data.zip")
    logger.info(f"importing zipped data {input_file}")

    app = pd.read_csv(os.path.join(input_filepath, "application_record.zip"))
    credit = pd.read_csv(os.path.join(input_filepath, "credit_record.zip"))

    data = app.merge(credit, on="ID")
    data = data[:30000]
    data['Male?'] = data["CODE_GENDER"].apply(lambda x: 1 if x == "M" else 0)
    data['Own Car?'] = data["FLAG_OWN_CAR"].apply(lambda x: 1 if x == "Y" else 0)
    data['Own Realty?'] = data["FLAG_OWN_REALTY"].apply(lambda x: 1 if x == "Y" else 0)
    data["Partnered?"] = data['NAME_FAMILY_STATUS'].apply(
        lambda x: 0 if x in ["Single / not married", "Widowed", "Separated"] else 1)
    data['Working?'] = data['NAME_INCOME_TYPE'].apply(lambda x: 0 if x in ["Pensioner", "Student"] else 1)
    data['Live with Parents?'] = data['NAME_HOUSING_TYPE'].apply(lambda x: 1 if x == "With parents" else 0)
    data['Days Old'] = data['DAYS_BIRTH'].apply(lambda x: -x)
    data = data[data['DAYS_EMPLOYED']<0]
    data['Days Employed'] = data['DAYS_EMPLOYED'].apply(lambda x: -x)

    data["Default?"] = data["STATUS"].apply(lambda x: 0 if x in ["C", "X"] else 1)
    data = data.drop(
        ["ID", "STATUS", "MONTHS_BALANCE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
         'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE', "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE",
         "FLAG_EMAIL", "OCCUPATION_TYPE", 'DAYS_BIRTH', "DAYS_EMPLOYED"], 1)
    data = data.rename(
        columns={"CNT_CHILDREN": "# Children", "AMT_INCOME_TOTAL": "Total Income", "DAYS_BIRTH": "Days Since Birth",
                 "DAYS_EMPLOYED": "Days Employed", "CNT_FAM_MEMBERS": "# Family Members"})

    debts = data[data["Default?"] == 1].index
    nodebts = data[data["Default?"] == 0][:len(debts)].index

    data = data.loc[[i for i in list(debts)+list(nodebts)]]

    data = data.reset_index(drop=True)
    outputs_dest_file = os.path.join(output_filepath, "inputs.csv")
    X = data.drop('Default?', axis=1)
    X.to_csv(outputs_dest_file, index=False)
    outputs_dest_file = os.path.join(output_filepath, "outputs.csv")
    Y = data[['Default?']]
    Y.to_csv(outputs_dest_file, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
