import sys
import os

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, thisdir)
sys.path.insert(0, os.path.join(thisdir, 'tsn'))
TSN_ROOT = os.path.join(thisdir, 'tsn')
