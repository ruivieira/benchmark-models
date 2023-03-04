# benchmark-models

A curated collection of datasets and models using the `cookiecutter` Data Science convention.

## Datasets

### Multi-output

- [pima-indians-diabetes-multi](./pima-indians-diabetes-multi), the Pima Indians diabetes dataset

### Model fairness

- [law-data](./law-data), the Law School Admission Council survey data.
- [credit-bias](./credit-bias), for credit bias analysis and exploration.
- [bias-loan](./bias-loan/), for a biased loan dataset and model.

## Models

| Name                                  | Inputs                 | Outputs       | Type                    |
|---------------------------------------|------------------------|---------------|-------------------------|
| [credit-bias](./credit-bias)          | -                      | Single output | -                       |
| [bias-loan](./bias-loan) | Numerical | Single output  | Random forest classifier |
| [pima-indians-diabetes-multi](./pima-indians-diabetes-multi) | Numerical | Multi output  | Decision Tree regressor |
