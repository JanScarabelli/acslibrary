# -*- coding: utf-8 -*-
"""
Created on June 22 19:56:28 2022

@author: WilliamNadolski

NAME:       logging_output_wrapper.py
PURPOSE:    reroute standard console output to external log file and capture unexpected exceptions
INPUTS:     external python script to capture log output for
OUTPUTS:    <program>_<dttm>.log = log containing all standard console output
            _ERROR_<program>_<dttm>.log = error log containing traceback for critical exceptions encountered
DEPENDENCY: none (except presence of pyfilepath code to execute)
PARAMS:     logDir = output directory to route logs to
            pyfilepath = input python code file to be executed within wrapper
EXCEPTIONS: none captured for this wrappeer itself, but will capture for program to be run/wrapped
LOGIC:      Wrapper will define output filepath for log based on program name and execution dttm.
            It then reroutes all standard console output to external log file.
            If unexpected exceptions encountered, it will also create an error log file with traceback info
REFERENCES: https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file
            https://towardsdatascience.com/building-and-exporting-python-logs-in-jupyter-notebooks-87b6d7a86c4
            https://stackoverflow.com/questions/6386698/how-to-write-to-a-file-using-the-logging-python-module/6386990#6386990

"""

#%% IMPORT NECESSARY PACKAGES

import os
import sys
import time
import calendar
import datetime
import logging

#%% ROUTE LOG OUTPUT

#define log dir and py file to execute
logDir = r"/Users/william.nadolski/Desktop/Python"
pyfilepath = r'/Users/william.nadolski/Desktop/Python/test.py'

#define file naming conventions
dttmUTC = datetime.datetime.utcnow()
UTCdttm = calendar.timegm(dttmUTC.utctimetuple())
program = pyfilepath.split(os.sep)[-1].replace('.py', '')
logFile = program + '_' + str(UTCdttm) + '.log'
logFilePath = os.path.join(logDir, logFile)

#try to exec code and capture stdoutput to log file
try:
    
    #save original stdout loc and open log for writing
    stdout_loc_orig = sys.stdout
    log = open(logFilePath, 'w')
    sys.stdout = log
    
    #insert python code file to execute here
    exec(open(pyfilepath).read())
    
    #reassign stdout loc and close log file
    sys.stdout = stdout_loc_orig
    log.close()
    
#if unexpected exception encountered then output to error log file
except Exception as e:
    errorLogFile = '_ERROR_' + logFile
    errorLogFilePath = os.path.join(logDir, errorLogFile)
    logging.basicConfig(filename=errorLogFilePath, level=logging.ERROR, filemode='w')
    logging.error(e, exc_info=True)

