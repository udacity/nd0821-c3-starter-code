#!/bin/bash

ENV_NAME=$1
COMMAND=$2

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
eval $COMMAND
conda deactivate