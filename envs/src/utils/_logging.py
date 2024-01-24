import sys
import os
from typing import Dict, List, Optional, TextIO, Union
import colorlog
from colorlog import ColoredFormatter
from colorlog.formatter import LogColors
import logging
from logging import Logger, Formatter
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL


DEFAULT_TIME_FORMAT = logging.Formatter.default_time_format
DEFAULT_LOG_COLORS = colorlog.default_log_colors
DEFAULT_LOG_FORMAT = "[%(levelname)s][%(name)s] %(asctime)s: %(message)s"
DEFAULT_ENCODING = "utf-8"

_SupportedLoggerType = Union[str, None, Logger]


def formatter_PlainText(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_TIME_FORMAT):
    return logging.Formatter(fmt, datefmt)


def formatter_ColorText(
    fmt=DEFAULT_LOG_FORMAT,
    datefmt=DEFAULT_TIME_FORMAT,
    log_colors: LogColors = DEFAULT_LOG_COLORS,
):
    return ColoredFormatter("%(log_color)s" + fmt, datefmt, log_colors=log_colors)


def _getLogger(x: _SupportedLoggerType):
    if x is None or isinstance(x, str):
        return logging.getLogger(x)
    if isinstance(x, Logger):
        return x
    raise ValueError(f"{x} is neither a logger nor a logger's name")


def _getLoggerName(x: _SupportedLoggerType):
    """root 不在 Logger.manager.loggerDict 中,而且用 'root' 创建的 logger 不是 root logger, 这是最让人混淆的"""
    raise NotImplementedError
    if x is None:
        return "root"
    if isinstance(x, Logger):
        return x.name
    if isinstance(x, str):
        return x
    return


def logger_clearHandlers(logger: _SupportedLoggerType, close=True):
    logger = _getLogger(logger)
    while len(logger.handlers):
        handler = logger.handlers[-1]
        if close:
            handler.close()
        logger.removeHandler(handler)
    return logger


def logger_addPlainTextFileHandler(
    logger: Logger,
    log_fn: str,
    level: int = None,  # 默认用 logger 的消息级别
    fmt=DEFAULT_LOG_FORMAT,
    datefmt=DEFAULT_TIME_FORMAT,
    append=True,
    encoding=DEFAULT_ENCODING,
):
    logger = _getLogger(logger)

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    os.makedirs(os.path.dirname(log_fn), exist_ok=True)
    file_handler = logging.FileHandler(
        log_fn, mode="a" if append else "w", encoding=encoding
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level if level is None else level)

    logger.addHandler(file_handler)
    return file_handler


def logger_addColorTextStreamHandler(
    logger: Logger,
    stream: TextIO = None,  # 默认输出到 sys.stderr
    level: int = None,  # 默认用 logger 的消息级别
    fmt=DEFAULT_LOG_FORMAT,
    datefmt=DEFAULT_TIME_FORMAT,
    log_colors: Dict[str, str] = DEFAULT_LOG_COLORS,
):
    logger = _getLogger(logger)

    formatter = formatter_ColorText(fmt, datefmt, log_colors)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logger.level if level is None else level)

    logger.addHandler(stream_handler)
    return stream_handler


def logger_quickReset(
    name: Optional[str],  # logger id, 若为空/空串则返回 root logger(不建议)
    level: Optional[int] = None,  # 默认不更改级别
    fmt=DEFAULT_LOG_FORMAT,
    #
    log_fn: Optional[str] = None,  # 输出 .log 文件,默认不输出到文件
    append=True,
    encoding=DEFAULT_ENCODING,  # 文本编码
    #
    log_colors: LogColors = DEFAULT_LOG_COLORS,  # 文本颜色(不应用到文件)
    to_stdout=False,  # 是否输出到标准输出流
    to_stderr=False,  # 是否输出到标准输出流
):
    # 获取/创建一个自定义的日志记录器
    logger = logging.getLogger(name)
    logger_clearHandlers(logger)

    import sys

    stdstreams: List[TextIO] = []
    if to_stdout:
        stdstreams.append(sys.stdout)
    if to_stderr:
        stdstreams.append(sys.stderr)
    for stream in stdstreams:
        logger_addColorTextStreamHandler(
            logger=name, stream=stream, fmt=fmt, log_colors=log_colors
        )

    if log_fn:
        log_fn = os.path.abspath(log_fn)
        logger_addPlainTextFileHandler(
            logger=name,
            log_fn=log_fn,
            fmt=fmt,
            append=append,
            encoding=encoding,
        )

    if level is not None:
        logger.setLevel(level)
    assert (
        logger.level != logging.NOTSET
    ), f"logger '{logger.name}'.level must not be NOTSET"
    for handler in logger.handlers:
        handler.setLevel(logger.level)
    return logger


import datetime


def logger_isRoot(logger: _SupportedLoggerType):
    if isinstance(logger, str) or logger is None:
        return not logger
    if isinstance(logger, Logger):
        return logger.root is logger
    raise TypeError(f"invalid logger type={type(logger)}")


def logger_isAncestor(parent: Logger, sub: Logger):
    """lhs is ancestor of rhs"""
    assert isinstance(parent, Logger), f"parent (type={type(parent)}) is not Logger"
    assert isinstance(sub, Logger), f"sub (type={type(sub)}) is not Logger"
    if parent is sub:
        return False
    if logger_isRoot(sub):
        return False
    if logger_isRoot(parent):
        return True
    return sub.name.startswith(parent.name + ".")


def logger_getSuffix(parent: Logger, sub: Logger):
    assert logger_isAncestor(
        parent, sub
    ), f"parent (name={parent.name}) is not ancestor of sub (name={sub.name})"
    if logger_isRoot(parent):
        return sub.name
    return sub.name[len(parent.name + ".") :]


def demo():
    logger1 = logger_quickReset("temp_logger", level=logging.DEBUG, to_stdout=True)
    logger2 = logger1.getChild("sub1")
    logger_quickReset(logger2.name, level=logging.INFO, to_stdout=False)
    logger3 = logger2.getChild("sub2")
    logger_quickReset(logger3.name, level=logging.WARNING, to_stdout=True)

    def test_logger(logger: Logger):
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.fatal("fatal")

    test_logger(logger1)
    print()
    test_logger(logger2)
    print()
    test_logger(logger3)


# 初始化 root logger(不建议)
# logger_quickReset(None, level=logging.INFO, to_stderr=True)

if __name__ == "__main__":
    demo()
