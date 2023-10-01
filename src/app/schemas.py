#!/usr/bin/env -S python3 -i

"""
Script to handle the FastAPI schema code.
author: Ilona Brinkmeier
date:   2023-09
"""

###################
# Imports
###################

from pydantic import BaseModel
from typing import Optional
from enum import Enum


###################
# Coding
###################

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
    age: int
    workclass: Optional[str] = None
    fnlgt: int
    education: Optional[str] = None
    education_num: int
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str] = None
