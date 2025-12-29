import os
from datetime import datetime

class SimLogger:
    _loggers = {}

    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    DEFAULT_LEVEL = "INFO"
    LOG_FILE = "logs/simulation.log"

    def __init__(self, module, level=None):
        self.module = module
        level = level or self.DEFAULT_LEVEL
        self.level = self.LEVELS[level]

        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

    @classmethod
    def get_logger(cls, module, level=None):
        if module not in cls._loggers:
            cls._loggers[module] = cls(module, level)
        return cls._loggers[module]

    def _format(self, level, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] [{level}] [{self.module}] {message}"

    def _log(self, level, message):
        if self.LEVELS[level] < self.level:
            return

        text = self._format(level, message)

        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        print(text)

    def debug(self, msg):
        self._log("DEBUG", msg)

    def info(self, msg):
        self._log("INFO", msg)

    def warning(self, msg):
        self._log("WARN", msg)

    def error(self, msg):
        self._log("ERROR", msg)
