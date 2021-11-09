# Model Card

For additional information see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf])

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
* precision: 0.6992  - is the fraction of relevant instances among the retrieved instance.
* recall: 0.28395 - is the fraction of relevant instances among the retrieved instance.
* fbeta: 0.40388 - is the weighted harmonic mean of precision and recall.

They are calculated using the [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) module.

## Ethical Considerations

The data set was donated to the UCI Machine Learning Repository in 1996. So the data points are more than 25 years old.
It is also important to be aware that salary in the data set is not a continuous value. But a binary value selected with a threshold of $50K/yr.
For the `native-country` filed there are a total of 40 countries represented and 583 records have null values. 
For inference on new data this should be taken into account on the models results.

## Caveats and Recommendations

I cannot guarantee that the model will perform well in a real world scenario. 
This is more of a demonstration of performing end-to-end model training and production deployment.