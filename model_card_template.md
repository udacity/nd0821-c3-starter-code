# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Type:** Random Forest Classifier  
**Developer:** Eric Koch  
**Date Created:** 2024-05-30  
**Frameworks and Libraries:** scikit-learn, pandas, numpy  
**Algorithm:** Ensemble method using multiple decision trees  
**Hyperparameters:** 
- n_estimators: 200
- max_depth: 30
- min_samples_split: 2
- min_samples_leaf: 1

## Intended Use

**Primary Use Case:** Predicting whether an individual's salary exceeds $50,000 based on census data.  
**Intended Users:** Data scientists, analysts, organizations  
**Use Cases:** Targeted advertising, demographic analysis, economic research  
**Limitations:** May not generalize well to future data if demographic patterns change.

## Training Data

**Data Source:** U.S. Census data  
**Data Size:** ~32,000 instances  
**Features:** Workclass, Education, Marital-status, Occupation, Relationship, Race, Sex, Native-country, Age, Education-num, Capital-gain, Capital-loss, Hours-per-week  
**Label:** Salary (binary: >50K or <=50K)

## Evaluation Data

**Data Source:** Subset of the U.S. Census data  
**Data Size:** ~6,400 instances (20% of the original dataset)  
**Features and Label:** Same as training data

## Metrics

**Evaluation Metrics:** 
- Precision: 0.913
- Recall: 0.734
- F-beta Score: 0.814

## Ethical Considerations

**Bias and Fairness:** Potential biases in historical data may affect predictions.  
**Privacy:** Ensure data privacy and compliance with relevant regulations.  
**Misuse Potential:** Avoid using the model for discriminatory purposes.

## Caveats and Recommendations

- **Generalization:** Model may not generalize well to future data.
- **Data Quality:** Accuracy depends on the quality of the input data.
- **Regular Updates:** Retrain periodically with updated data.
- **Interpretability:** Use feature importance scores to interpret the model.
- **Ethical Use:** Ensure fair and responsible use of the model.