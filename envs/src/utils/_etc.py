import math
from typing import Iterable, List, MutableSequence, Sequence, Union

import re
import datetime

import pandas as pd


def parse_optional(val, default_=None):
    return default_ if val is None else val


def datetime2str(dt: datetime.datetime = None, fmt="%Y%m%d_%H%M%S"):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime(fmt)


__pattern_invalid_ros_node_name = re.compile("[^\w]", re.I)


def ros_valid_node_id(s: str):
    rst = "node" + s
    rst = "x" + __pattern_invalid_ros_node_name.sub("_", rst)
    return rst
