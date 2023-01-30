from pydantic import BaseModel
from typing_extensions import Literal


class ModelInput(BaseModel):
    # https://archive.ics.uci.edu/ml/datasets/census+income
    age: int

    workclass: Literal[
        'Private',
        'Self-emp-not-inc',
        'Self-emp-inc',
        'Federal-gov',
        'Local-gov',
        'State-gov',
        'Without-pay',
        'Never-worked',
    ]

    fnlwgt: int

    education: Literal[
        'Bachelors',
        'Some-college',
        '11th',
        'HS-grad',
        'Prof-school',
        'Assoc-acdm',
        'Assoc-voc',
        '9th',
        '7th-8th',
        '12th',
        'Masters',
        '1st-4th',
        '10th',
        'Doctorate',
        '5th-6th',
        'Preschool',
    ]

    education_num: int

    marital_status: Literal[
        'Married-civ-spouse',
        'Divorced',
        'Never-married',
        'Separated',
        'Widowed',
        'Married-spouse-absent',
        'Married-AF-spouse',
    ]

    occupation: Literal[
        'Tech-support',
        'Craft-repair',
        'Other-service',
        'Sales',
        'Exec-managerial',
        'Prof-specialty',
        'Handlers-cleaners',
        'Machine-op-inspct',
        'Adm-clerical',
        'Farming-fishing',
        'Transport-moving',
        'Priv-house-serv',
        'Protective-serv',
        'Armed-Forces',
    ]

    relationship: Literal[
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ]

    race: Literal['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']

    sex: Literal['Female', 'Male']

    capital_gain: int

    capital_loss: int

    hours_per_week: int

    native_country: Literal[
        'United-States',
        'Cambodia',
        'England',
        'Puerto-Rico',
        'Canada',
        'Germany',
        'Outlying-US(Guam-USVI-etc)',
        'India',
        'Japan',
        'Greece',
        'South',
        'China',
        'Cuba',
        'Iran',
        'Honduras',
        'Philippines',
        'Italy',
        'Poland',
        'Jamaica',
        'Vietnam',
        'Mexico',
        'Portugal',
        'Ireland',
        'France',
        'Dominican-Republic',
        'Laos',
        'Ecuador',
        'Taiwan',
        'Haiti',
        'Columbia',
        'Hungary',
        'Guatemala',
        'Nicaragua',
        'Scotland',
        'Thailand',
        'Yugoslavia',
        'El-Salvador',
        'Trinadad&Tobago',
        'Peru',
        'Hong',
        'Holand-Netherlands',
    ]

    class Config:
        # https://fastapi.tiangolo.com/tutorial/schema-extra-example/
        schema_extra = {
            "example": {
                'age': 25,
                'workclass': 'Self-emp-not-inc',
                'fnlwgt': 176756,
                'education': 'HS-grad',
                'education_num': 9,
                'marital_status': 'Never-married',
                'occupation': 'Farming-fishing',
                'relationship': 'Own-child',
                'race': 'White',
                'sex': 'Male',
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 35,
                'native_country': 'United-States',
            }
        }
