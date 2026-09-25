# utils/logger.py
import logging
import sys


_initialized = False


def setup_root_logger(level=logging.INFO):
    """全局日志配置，只在入口（main.py）调用一次"""
    global _initialized
    if _initialized:
        return

    root = logging.getLogger("geeked")
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)

    _initialized = True


def get_logger(name):
    """
    获取模块级 logger。
    统一加上 geeked 前缀，便于统一控制级别。
    """
    if not name.startswith("geeked"):
        name = f"geeked.{name}"
    return logging.getLogger(name)


def silence_geeked(level=logging.WARNING):
    """
    静默 geeked 的日志，供测试脚本使用。
    只显示 WARNING 及以上的日志。
    """
    logging.getLogger("geeked").setLevel(level)