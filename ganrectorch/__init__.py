__author__ = """Xiaogang Yang"""
__email__ = "yangxg@bnl.gov"
__version__ = "0.1.0"

from ganrectorch.ganrec import *
from ganrectorch.utils import *
from ganrectorch.models import *
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
