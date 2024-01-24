from . import _path
from ._path import (
    path_parent,
    path_find,
    path_find_with_interaction,
)

from . import _math
from ._math import (
    init_seed,
    affine_comb,
    mod_range,
    rewards2gain,
    SmoothAve,
)
from .geom_ import *
from . import _etc
from ._etc import (
    parse_optional,
    ros_valid_node_id,
    datetime2str,
)

from . import _logging
from ._logging import (
    Logger,
    logger_addColorTextStreamHandler,
    logger_addPlainTextFileHandler,
    logger_clearHandlers,
    logger_isAncestor,
    logger_isRoot,
    logger_getSuffix,
    logger_quickReset,
    Formatter,
    ColoredFormatter,
    formatter_ColorText,
    formatter_PlainText,
)


def show_all_loggers():
    import logging

    manager = logging.Logger.manager
    logger_names = manager.loggerDict.keys()
    msg = "loggers(except root)=" + np.asarray(logger_names)
    print(msg)


if __name__ == "__main__":
    raise Exception(f"不允许直接运行 {__file__}")
