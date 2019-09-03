import sys
import os


def addpath(dir):
    if dir not in sys.path:
        sys.path.insert(0, dir)

THIS_DIR = os.path.dirname(__file__)
API_TSN_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))

addpath(API_TSN_DIR)
