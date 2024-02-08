#!/usr/bin/env -S python3 -i

"""
Script to handle the FastAPI schema code.

author: Ilona Brinkmeier
date:   2023-09

update 2024-02:
For usage of Pydantic V2 as FastAPI dependency see:
https://docs.pydantic.dev/latest/concepts/json_schema/
"""

###################
# Imports
###################

from pydantic import ConfigDict, BaseModel, Field
from enum import Enum


###################
# Coding
###################

def hyphen_to_underscore(field_name):
    ''' Replaces hyphen with underscore '''
    return f"{field_name}".replace("-", "_")


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
    ''' Delivers the column label handling for feature attributes '''
    age: int = Field(..., examples=[45])
    capital_gain: int = Field(..., examples=[2174])
    capital_loss: int = Field(..., examples=[0])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(..., examples=[13])
    fnlgt: int = Field(..., examples=[2334])
    hours_per_week: int = Field(..., examples=[60])
    marital_status: str = Field(..., examples=["Never-married"])
    native_country: str = Field(..., examples=["Cuba"])
    occupation: str = Field(..., examples=["Prof-specialty"])
    race: str = Field(..., examples=["Black"])
    relationship: str = Field(..., examples=["Wife"])
    sex: str = Field(..., examples=["Female"])
    workclass: str = Field(..., examples=["State-gov"])
    model_config = ConfigDict(alias_generator=hyphen_to_underscore, populate_by_name=True)
