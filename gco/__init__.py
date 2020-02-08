__version__ = '3.0.2'
__version_info__ = (3, 0, 2)

# import cgco
# import pygco

try:
    from pygco import *
except:
    from gco.pygco import *