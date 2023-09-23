#!/usr/bin/env -S python3 -i

"""
Script to handle configuration settings used in the ML pipeline.
author: Ilona Brinkmeier
date:   2023-09
"""

###################
# Imports
###################
import logging
import yaml
import os
import sys

###################
# Coding
###################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)


def get_project_root_path():
    '''  Returns the absolute path to projects root. '''
    return os.getcwd()


def create_config():
    ''' Reads in config file. '''
    ROOT = get_project_root_path()
    CONFIG_PATH = 'src/config/config.yml'
    # add "src" dir, source code and config is there
    CONFIG_FILE = os.path.join(ROOT, CONFIG_PATH)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                config_dict = yaml.safe_load(f.read())
                logger.info('Configuration yml file content is:\n %s', config_dict)
                return config_dict
            except yaml.YAMLError:
                logger.exception("Exit: Error parsing config.yml in config.__init__.py.")
                sys.exit(1)
    else:
        logger.exception("Exit: Configuration file does not exist on path: %s", CONFIG_FILE)
        sys.exit(1)


def get_config():
    ''' Returns dictionary about project configuration. '''
    # future toDo: create config class with _init_ to have config file once
    config_dict = create_config()
    return config_dict


def get_data_path():
    ''' Returns the data directory path. '''
    ROOT = get_project_root_path()
    logger.info('config.py: ROOT: %s for data storage dir', ROOT)
    config_dict = get_config()
    data_path = os.path.join(ROOT, config_dict['etl']['data_path'])
    logger.info("config data_path: %s", data_path)
    return data_path


def get_models_path():
    ''' Returns the models directory path. '''
    ROOT = get_project_root_path()
    logger.info('config.py: ROOT: %s for models storage dir', ROOT)
    config_dict = get_config()
    models_path = os.path.join(ROOT, config_dict['model']['models_path'])
    logger.info("config models_path: %s", models_path)
    return models_path
