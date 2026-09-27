import threading

import numpy as np

from neurale.devices import Signal


class ExampleDevice:
    def open(self, config):
        self.config = config
        self.thread = None
        self.stop = threading.Event()

    def describe(self):
        return (
            Signal("amplifier", 2, 1000, 4),
            Signal("auxiliary", 1, 500, 2),
            Signal("trigger", 1, 0, 1, dtype="int32", kind="event", unit="dimensionless"),
        )

    def start(self, ingress):
        def acquire():
            try:
                if self.config.get("crash"):
                    import os

                    os._exit(7)
                if self.config.get("stall"):
                    self.stop.wait()
                    return
                ingress.publish(0, np.arange(8, dtype=np.float64).reshape(4, 2))
                ingress.gap(0, metadata={"missing_samples": 2})
                ingress.publish(0, np.arange(12, 16, dtype=np.float64).reshape(2, 2))
                ingress.publish(1, np.array([[20.0], [21.0]]))
                ingress.publish(2, np.array([[7]], dtype=np.int32))
                ingress.gap(0, restart=True)
                ingress.finish()
            except Exception:
                if not ingress.cancelled:
                    ingress.fail()

        self.thread = threading.Thread(target=acquire)
        self.thread.start()

    def cancel(self):
        self.stop.set()

    def close(self):
        self.cancel()
        if self.thread is not None:
            self.thread.join()
