"""Importable fake Python SDKs for process lifecycle tests."""

import time
from pathlib import Path

from neurale.devices import Signal


class ControlledProvider:
    def open(self, config):
        self.config = config
        self.channels = 1
        if "lifecycle_log" in config:
            self._record("open")
        if config.get("fail_open"):
            raise RuntimeError("SDK refused connection")
        if config.get("slow_open"):
            time.sleep(30)
        if "schema_marker" in config:
            marker = Path(config["schema_marker"])
            self.channels = 2 if marker.exists() else 1
            marker.touch()

    def describe(self):
        return [Signal("signal", self.channels, 1000, 1)]

    def start(self, ingress):
        if self.config.get("fail_start"):
            raise RuntimeError("SDK refused start")
        ingress.finish()

    def cancel(self):
        self._record("cancel")
        if self.config.get("fail_cancel"):
            raise RuntimeError("SDK refused cancel")

    def close(self):
        self._record("close")
        if self.config.get("exit_close"):
            import os

            os._exit(9)
        if self.config.get("fail_close"):
            raise RuntimeError("SDK refused close")
        if self.config.get("slow_close"):
            time.sleep(30)

    def _record(self, operation):
        if "lifecycle_log" in self.config:
            with Path(self.config["lifecycle_log"]).open("a") as log:
                log.write(operation + "\n")
