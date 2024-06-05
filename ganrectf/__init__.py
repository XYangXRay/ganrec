__author__ = """Xiaogang Yang"""
__email__ = "yangxg@bnl.gov"
__version__ = "0.1.0"

from ganrectf.ganrec import *
from ganrectf.utils import *
from ganrectf.models import *
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import _version
__version__ = _version.get_versions()['version']
