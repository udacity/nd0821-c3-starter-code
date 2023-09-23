#!/usr/bin/env -S python3 -i

"""
Script to handle the API code here.
author: Ilona Brinkmeier
date:   2023-09
"""

###################
# Imports
###################

import logging
import uvicorn


###################
# Coding
###################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)







if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
