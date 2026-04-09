from enum import IntEnum
from inspect import stack
import os

TEST_FLAG = False
class LogLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3

class Log(object):
    c_current_log_level = LogLevel.INFO

    @staticmethod
    def get_info():
        s = stack()
        line = s[2].lineno
        path = s[2].filename
        filename = os.path.split(path)[-1]
        return "file: " + filename + ": line: " + str(line) + ": "

    @staticmethod
    def write_to_file(msg):
        if not os.path.isfile("output/logs.txt"):
            open("output/logs.txt", "w").close()
        with open("output/logs.txt", "a+") as f:
            f.write(msg + '\n')

    @staticmethod
    def print_error(msg):
        info = Log.get_info()
        if Log.c_current_log_level <= LogLevel.ERROR:
           Log.write_to_file("ERROR: " + info + msg)

    @staticmethod
    def print_info(msg):
        info = Log.get_info()
        if Log.c_current_log_level <= LogLevel.INFO:
            Log.write_to_file("INFO: " + info + msg)

    @staticmethod
    def print_warning(msg):
        info = Log.get_info()
        if Log.c_current_log_level <= LogLevel.WARNING:
            Log.write_to_file("WARNING: " + info + msg)