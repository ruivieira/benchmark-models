# `credit-bias` model

Data for credit bias analysis and exploration.

- Data originally from the _Bondora_ loan dataset [^1]
- Data processing based on the _Bias in credit models_ code [^2]

[^1]: https://www.bondora.com/en/public-reports
[^2]: https://github.com/valeria-io/bias-in-credit-models

## Dataset creation

Use 

``` shell
$ make features
```
to generate a training dataset `data/processed/train.csv`.

If you are only interested in a pre-processed dataset (`data/interim/data.csv`), use

``` shell
$ make data
```

## Model training

To train a XGBoost model use

```shell
$ make train
```

The model will be serialised as a JSON XGBoost by default. This output format can be changed in the `Makefile`
by specifying one of the following values:

- `ubj`, Universal Binary JSON
- `json`, JSON
- `pmml`, PMML

## Jupyter notebooks

- [Data exploration](./notebooks/01-data-exploration.ipynb)