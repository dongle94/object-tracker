import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import os
import sys
from pathlib import Path
from datetime import datetime


logger = None


def init_logger(cfg=None, name="default", loglevel="debug"):
    # LOG FORMATTING
    # https://docs.python.org/ko/3.8/library/logging.html#logrecord-attributes
    log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
    timestamp = datetime.now().strftime("%Y%m%d")

    if cfg is not None:
        # LOGGER NAME
        name = cfg.logger_name if cfg.logger_name else name
        _logger = logging.getLogger(name)
        if _logger.hasHandlers():
            _logger.handlers.clear()

        # LOG LEVEL
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = cfg.log_level.upper() if cfg.log_level else loglevel.upper()
        if log_level not in valid_log_levels:
            raise ValueError(f"Invalid log level: {log_level}")
        _logger.setLevel(log_level)

        # LOG CONSOLE
        if cfg.console_log is True:
            _handler = StreamHandler()
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

        # LOG FILE
        if cfg.file_log is True:
            filename = os.path.join(cfg.file_log_dir, f"{cfg.logger_name}_{timestamp}" + '.log')
            logdir = os.path.dirname(filename)
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            when = cfg.file_log_rotate_time if hasattr(cfg, "file_log_rotate_time") else "D"
            interval = cfg.file_log_rotate_interval if hasattr(cfg, "file_log_rotate_interval") else 1
            _handler = TimedRotatingFileHandler(
                filename=filename,
                when=when,
                interval=interval,
                backupCount=cfg.file_log_counter,
                encoding='utf8'
            )
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

    else:       # cfg is None
        loglevel = loglevel.upper()
        _logger = logging.getLogger(name)
        if _logger.hasHandlers():
            _logger.handlers.clear()
        _logger.setLevel(loglevel)

        # CONSOLE LOGGER
        _handler = StreamHandler()
        _handler.setLevel(loglevel)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

        # FILE LOGGER
        filename = os.path.join('./log', f"{name}_{timestamp}" + '.log')
        logdir = os.path.dirname(filename)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        _handler = TimedRotatingFileHandler(
            filename=filename,
            when="D",
            interval=1,
            backupCount=10,
            encoding='utf8'
        )
        _handler.setLevel(loglevel)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

    _logger.info("Start Main logger")
    global logger
    logger = _logger


def get_logger(name=None):
    if name is None:
        global logger
        return logger
    else:
        return logging.getLogger(name)


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    os.chdir(ROOT)  # set the current path is ROOT

    init_logger()
    get_logger().info("hello world")
