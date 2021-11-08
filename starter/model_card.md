# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The trained model is a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) that has been trained to predict whether income exceeds $50K/yr based on census data.

## Intended Use

The model could be used as a baseline to compare against other models.

## Training Data

Uses the UCI Machine Learning Repository data set [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) and all 14 features.

## Evaluation Data

20% of the data is used for evaluation.

## Metrics

The following metrics are used to evaluate the models predictions:
* precision - is the fraction of relevant instances among the retrieved instance.
* recall - is the fraction of relevant instances among the retrieved instance.
* fbeta - is the weighted harmonic mean of precision and recall.

They are calculated using the [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) module.

## Ethical Considerations

The data set was donated to the UCI Machine Learning Repository 1996-05-01. So the data points are more than 25 years old.

## Caveats and Recommendations

I cannot guarantee that the model will perform well in a real world scenario. 
This is more of a demonstration of a simple end-to-end model training and production deployment.