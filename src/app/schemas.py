#!/usr/bin/env -S python3 -i

"""
Script to handle the FastAPI schema code.
author: Ilona Brinkmeier
date:   2023-09
"""

###################
# Imports
###################

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


###################
# Coding
###################

def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")

class FeatureLabels(str, Enum):
    ''' Delivers the feature names as needed in Python '''
    age = "age"
    workclass = "workclass"
    fnlgt = "fnlgt"
    education = "education"
    education_num = "education_num"
    marital_status = "marital_status"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    captial_gain = "capital_gain"
    captial_loss = "capital_loss"
    hours_per_week = "hours_per_week"
    native_country = "native_country"


class Person(BaseModel):
    ''' Delivers the type hints for feature attributes '''
#    age: int
#    workclass: Optional[str] = None
#    fnlgt: int
#    education: Optional[str] = None
#    education_num: int
#    marital_status: Optional[str] = None
#    occupation: Optional[str] = None
#    relationship: Optional[str] = None
#    race: Optional[str] = None
#    sex: Optional[str] = None
#    capital_gain: int
#    capital_loss: int
#    hours_per_week: int
#    native_country: Optional[str] = None

    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True
