# Model Card

Additional information can be found on arvix: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- The model is a business developed in Random Forest to forecast contracts. The classification in this case is binary, a sample can be positive (1) or negative (0).

## Intended Use
- Project developed for studies and personal development.

## Training Data
- The data utilized for training this model came from the Census Beurau, and consists of salary information: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
- Evaluation data comes from the same dataset, just a 20% split from dataset

## Metrics

```python
Precision: 0.7421
Recall: 0.6303
F1: 0.66817
```

## Ethical Considerations
- All models can have ethnic, racial and cultural biases. These aspects need to be carefully observed when processing the data.

## Caveats and Recommendations
- As it includes more data pre-processing and feature engineering and data analysis to improve results.
- How many ethnic studies, further study and analysis of data from other geographic aspects is needed as well.
