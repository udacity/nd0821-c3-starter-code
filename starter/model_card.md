# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- This model is created for the ML DevOps Nanodegree from Udacity
- Created on 5/14/2021
- Version 1.0
- We used a Random Forest Classifier in this project
- For more information about the model, please refer to Udacity Nanodegree program


## Intended Use
- This project is intended to show an example of deploying a Scalable ML Pipeline in Production
- Predict whether income exceeds $50K/yr based on census data


## Factors
- This data is very old as it was collected in 1994, so we can't rely on it only in 2021
- We should consider the sampling groups as well

## Metrics
We have used three metrics in this project; fbeta_score, precision_score and recall_score.

## Evaluation Data
We have used 20% of the original dataset for evaluation purposes of the model.

## Training Data
We have used 80% of the original dataset for the training purposes of the model.

## Quantitative Analyses
Model performance is measured based on three metrics: fbeta score, precision score and recall score.
