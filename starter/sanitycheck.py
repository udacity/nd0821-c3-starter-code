from os import path

import argparse
import importlib
import inspect
import os
import sys

FAIL_COLOR = '\033[91m'
OK_COLOR = '\033[92m'
WARN_COLOR = '\033[93m'

def run_sanity_check(test_dir):

    #assert path.isdir(test_dir), FAIL_COLOR+f"No direcotry named {test_dir} found in {os.getcwd()}"
    print('This script will perform a sanity test to ensure your code meets the criteria in the rubric.\n')
    print('Please enter the path to the file that contains your test cases for the GET() and POST() methods')
    print('The path should be something like abc/def/test_xyz.py')
    filepath = input('> ')

    assert path.exists(filepath), f"File {filepath} does not exist."
    sys.path.append(path.dirname(filepath))

    module_name = path.splitext(path.basename(filepath))[0]
    module = importlib.import_module(module_name)


    test_function_names = list(filter(lambda x: inspect.isfunction(getattr(module,x)) and not x.startswith('__'), dir(module)))

    test_functions_for_get = list(filter(lambda x: inspect.getsource(getattr(module,x)).find('.get(') != -1 , test_function_names))
    test_functions_for_post = list(filter(lambda x: inspect.getsource(getattr(module,x)).find('.post(') != -1, test_function_names))
    

    print("\n============= Sanity Check Report ===========")
    SANITY_TEST_PASSING = True
    WARNING_COUNT = 1

    ## GET()
    TEST_FOR_GET_METHOD_RESPONSE_CODE = False
    TEST_FOR_GET_METHOD_RESPONSE_BODY = False
    if not test_functions_for_get:
        print(FAIL_COLOR+f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR+"No test cases were detected for the GET() method.")
        print(FAIL_COLOR+"\nPlease make sure you have a test case for the GET method.\
            This MUST test both the status code as well as the contents of the request object.\n")
        SANITY_TEST_PASSING = False

    else:
        for func in test_functions_for_get:
            source = inspect.getsource(getattr(module,func))
            if source.find('.status_code') != -1:
                TEST_FOR_GET_METHOD_RESPONSE_CODE = True
            if (source.find('.json') != -1) or (source.find('json.loads') != -1):
                TEST_FOR_GET_METHOD_RESPONSE_BODY =  True


        if not TEST_FOR_GET_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR+f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"Your test case for GET() does not seem to be testing the response code.\n")
        
        if not TEST_FOR_GET_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR+f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"Your test case for GET() does not seem to be testing the CONTENTS of the response.\n")



    ## POST() 
    TEST_FOR_POST_METHOD_RESPONSE_CODE = False
    TEST_FOR_POST_METHOD_RESPONSE_BODY = False
    COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT = 0

    if not test_functions_for_post:
        print(FAIL_COLOR+f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR+"No test cases were detected for the POST() method.")
        print(FAIL_COLOR+"Please make sure you have TWO test cases for the POST() method."+
        "\nOne test case for EACH of the possible inferences (results/outputs) of the ML model.\n")
        SANITY_TEST_PASSING = False
    else:
        if len(test_functions_for_post) == 1:
            print(f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"Only one test case was detected for the POST() method.")
            print(FAIL_COLOR+"Please make sure you have two test cases for the POST() method."+
            "\nOne test case for EACH of the possible inferences (results/outputs) of the ML model.\n")
            SANITY_TEST_PASSING = False

        for func in test_functions_for_post:
            source = inspect.getsource(getattr(module,func))
            if source.find('.status_code') != -1:
                TEST_FOR_POST_METHOD_RESPONSE_CODE = True
            if (source.find('.json') != -1) or (source.find('json.loads') != -1):
                TEST_FOR_POST_METHOD_RESPONSE_BODY =  True
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT += 1

        if not TEST_FOR_POST_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR+f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"One or more of your test cases for POST() do not seem to be testing the response code.\n")
        if not TEST_FOR_POST_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR+f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"One or more of your test cases for POST() do not seem to be testing the contents of the response.\n")

        if len(test_functions_for_post) >= 2 and COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT < 2:
            print(FAIL_COLOR+f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(FAIL_COLOR+"You do not seem to have TWO separate test cases, one for each possible prediction that your model can make.")



    SANITY_TEST_PASSING = SANITY_TEST_PASSING and\
        TEST_FOR_GET_METHOD_RESPONSE_CODE and \
        TEST_FOR_GET_METHOD_RESPONSE_BODY and \
        TEST_FOR_POST_METHOD_RESPONSE_CODE and \
        TEST_FOR_POST_METHOD_RESPONSE_BODY and \
        COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT >= 2

    if SANITY_TEST_PASSING:
        print(OK_COLOR+"Your test cases look good!")
    
    print(WARN_COLOR+"This is a heuristic based sanity testing and cannot guarantee the correctness of your code.")
    print(WARN_COLOR+"You should still check your work against the rubric to ensure you meet the criteria.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir',metavar='test_dir',nargs='?',default='tests',help='Name of the directory that has test files.')
    args = parser.parse_args()
    run_sanity_check(args.test_dir)

