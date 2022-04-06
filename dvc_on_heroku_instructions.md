# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model consists of a Random Forest classifier which predicts whether someones salary is above $50,000. 
For this use case a binary classification approach was taken, whereby a sample showing a probability of 1.0 being positive and 0.0 being negative case.

## Intended Use
- This model isn't inteded to be used in a production setting. Just for this udacity projects :)

## Training Data
- The data utilized for training this model came from the <b>Census Beurau</b>, and consists of salary information.
Dataset: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
- Evaluation data comes from the same dataset, a 20% split of the samples that are not used during train.

## Metrics

```python
Precision: 0.7421
Recall: 0.6303
F1: 0.6817
```