# SPDX-License-Identifier: MIT
from __future__ import annotations

import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from neurale import devices
from neurale._native_loader import load_native_namespace
from neurale.devices.provider import _native_signals
from neurale.streaming import _native as n


def config():
    budget = n.PoolCapacityBudget()
    budget.source_owned = 1
    budget.ingress_capacity = 16
    budget.processor_owned = 2
    budget.critical_edge_capacity = 17
    budget.actuator_owned = 1
    value = n.RealtimeConfig()
    value.pool_capacity = budget
    value.buffer_size = 128
    value.max_signal_blocks = 3
    value.discontinuity_capacity = 8
    value.gaps_per_discontinuity = 3
    value.max_process_outputs = 1
    value.max_flush_outputs = 0
    value.fault_history_capacity = 8
    return value


class Capture:
    def __init__(self):
        self.frames = []
        self.gaps = []

    def write(self, frame):
        self.frames.append(frame)

    def handle_discontinuity(self, gap):
        self.gaps.append(gap)


def run(source):
    capture = Capture()
    sink = n.PythonSinkAdapter(capture)
    processor = n.IdentityNativeProcessor()
    runner = n._NativeStreamRunner(source.schema, config(), source, processor, sink)
    assert runner.prepare() == n.StreamStatus.OK
    assert runner.arm() == n.StreamStatus.OK
    assert runner.run() == n.StreamStatus.OK
    return capture


def test_array_contract_and_ordered_gap(tmp_path):
    native = load_native_namespace("devices")
    signals = _native_signals([devices.Signal("test", 2, 1000, 4)])
    source = native.GenericDeviceSource(signals, str(tmp_path / "queue"), 16)
    ingress = source.ingress
    data = np.arange(8, dtype=np.float64).reshape(4, 2)
    with pytest.raises(ValueError, match="dtype"):
        ingress.publish(0, data.astype(np.float32))
    with pytest.raises(ValueError, match="contiguous"):
        ingress.publish(0, data[:, ::-1])
    ingress.publish(0, data, {"sample_index": 10, "device_tick": 100})
    ingress.gap(0, metadata={"missing_samples": 3})
    ingress.publish(0, data[:2])
    ingress.gap(0)
    ingress.finish()
    result = run(source)
    assert [f.blocks[0].sample_idx_start for f in result.frames] == [10, 17]
    np.testing.assert_array_equal(result.frames[0].payload.view(np.float64), data.ravel())
    assert [g.signal_gaps[0].missing_samples for g in result.gaps] == [3, None]
    source.close()


def test_queue_fault_is_explicit(tmp_path):
    native = load_native_namespace("devices")
    source = native.GenericDeviceSource(
        _native_signals([devices.Signal("test", 1, 1000, 1)]), str(tmp_path / "queue"), 2
    )
    ingress = source.ingress
    ingress.publish(0, np.ones((1, 1)))
    ingress.publish(0, np.ones((1, 1)))
    with pytest.raises(RuntimeError, match="rejected"):
        ingress.publish(0, np.ones((1, 1)))
    assert ingress.stats["error"] != 0
    source.close()


def test_declared_layout_ticks_and_clock_mapping(tmp_path):
    native = load_native_namespace("devices")
    source = native.GenericDeviceSource(
        _native_signals(
            [
                devices.Signal(
                    "test", 2, 1000, 3, dtype="int16", layout="channel_major", sample_counter=True
                )
            ]
        ),
        str(tmp_path / "queue"),
        8,
    )
    values = np.arange(6, dtype=np.int16).reshape(2, 3)
    ingress = source.ingress
    ingress.publish(
        0,
        values,
        {
            "device_tick": 100,
            "tick_reference": 100,
            "host_reference_ns": 10000,
            "tick_rate_numerator": 1000,
            "tick_rate_denominator": 1,
            "uncertainty_ns": 20,
            "clock_generation": 2,
            "synchronized": 1,
        },
    )
    ingress.finish()
    result = run(source)
    block = result.frames[0].blocks[0]
    assert block.device_tick_start == 100
    assert block.clock_sync.generation == 2
    assert block.clock_sync.uncertainty_ns == 20
    assert block.clock_sync.host_time_reference_ns == 10000
    np.testing.assert_array_equal(result.frames[0].payload.view(np.int16), values.ravel())
    source.close()


def test_missing_tick_fails_declared_sample_counter(tmp_path):
    native = load_native_namespace("devices")
    source = native.GenericDeviceSource(
        _native_signals([devices.Signal("test", 1, 1000, 1, sample_counter=True)]),
        str(tmp_path / "queue"),
        8,
    )
    with pytest.raises(RuntimeError, match="rejected"):
        source.ingress.publish(0, np.ones((1, 1)))
    assert source.ingress.stats["error"] != 0
    source.close()


def test_discovery_does_not_load_plugins(monkeypatch):
    from neurale.devices import _registry

    class Entry:
        name = "example.test"

        def load(self):
            raise AssertionError("discovery must not load the SDK")

    monkeypatch.setattr(_registry, "entry_points", lambda **kw: [Entry()])
    assert devices.list() == ("example.test",)


@pytest.mark.parametrize("kind", ["pull", "callback", "python"])
def test_installed_example(kind):
    if importlib.util.find_spec(f"neurale_example_{kind}") is None:
        pytest.skip("install examples/device_providers packages to test independent plugins")
    with devices.open(f"example.{kind}") as device:
        source = device.source
        consumer = n.CountingNativeConsumer()
        processor = n.IdentityNativeProcessor()
        safety = n.RecordingNativeSafetyController()
        runner = n._NativeStreamRunner(device.schema, config(), source, processor, consumer, safety)
        assert runner.prepare() == n.StreamStatus.OK
        assert runner.arm() == n.StreamStatus.OK
        device.start()
        assert runner.run() == n.StreamStatus.OK
        assert consumer.frame_count == 4
        assert consumer.discontinuity_count == 2
        assert runner.reset() == n.StreamStatus.OK
        assert runner.prepare() == n.StreamStatus.OK
        assert runner.arm() == n.StreamStatus.OK
        device.start()
        assert runner.run() == n.StreamStatus.OK
        assert consumer.frame_count == 4
        directory = Path(device._directory)
        runner.stop()
    del runner, source
    gc.collect()
    assert not directory.exists()


def test_python_process_cancellation_and_crash():
    if importlib.util.find_spec("neurale_example_python") is None:
        pytest.skip("install the Python example provider")
    with devices.open("example.python", stall=True) as device:
        device.start()
        started = time.monotonic()
        device.cancel()
        device.close()
        assert time.monotonic() - started < 5
    device = devices.open("example.python", crash=True)
    try:
        try:
            device.start()
        except RuntimeError:
            pass  # Crash may precede the start acknowledgement.
        deadline = time.monotonic() + 5
        while not device.stats["error"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert device.stats["error"] != 0
    finally:
        if device._process is not None:
            with pytest.raises(RuntimeError, match="exited with code"):
                device.close()
        else:
            device.close()  # start() already reported and reaped the failed process.


@pytest.fixture
def process_factory(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return devices.PythonProvider("provider_fixture:ControlledProvider")


def test_process_schema_change_refused(process_factory, tmp_path):
    with devices.Device(
        process_factory, config={"schema_marker": str(tmp_path / "opened")}
    ) as device:
        original = device.schema.signals[0].n_channels
        assert device.source.reset() == n.StreamStatus.SOURCE_FAILURE
        assert device.schema.signals[0].n_channels == original
        assert device._process is None


@pytest.mark.parametrize("setting", ["fail_open", "slow_open"])
def test_process_open_failure_cleanup(process_factory, setting):
    import multiprocessing

    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(RuntimeError):
        devices.Device(process_factory, config={setting: True}, timeout=0.5)
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_process_start_failure_cleanup(process_factory):
    with devices.Device(process_factory, config={"fail_start": True}) as device:
        with pytest.raises(RuntimeError, match="refused start"):
            device.start()
        assert device.stats["error"] != 0
        assert device._process is None


def test_process_hung_close_is_explicit(process_factory):
    from neurale.exceptions import StreamTimeoutError

    device = devices.Device(process_factory, config={"slow_close": True}, timeout=1.0)
    directory = Path(device._directory)
    device.start()
    with pytest.raises(StreamTimeoutError, match="terminated"):
        device.close()
    device.close()
    assert device.stats["error"] != 0
    gc.collect()
    assert not directory.exists()


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("failure", ["fail_cancel", "fail_close", "both", "exit_close"])
def test_process_shutdown_failure_reported_after_cleanup(
    process_factory, tmp_path, started, failure
):
    from neurale.exceptions import StreamExecutionError

    log = tmp_path / "lifecycle"
    options = {"lifecycle_log": str(log), failure: True}
    if failure == "both":
        options.update(fail_cancel=True, fail_close=True)
    device = devices.Device(process_factory, config=options)
    directory = Path(device._directory)
    if started:
        device.start()
    with pytest.raises(StreamExecutionError) as raised:
        device.close()
    message = str(raised.value)
    if failure in ("fail_cancel", "both"):
        assert "SDK refused cancel" in message
    if failure in ("fail_close", "both"):
        assert "SDK refused close" in message
    if failure == "exit_close":
        assert "exited with code 9" in message
    assert log.read_text().splitlines() == ["open", "cancel", "close"]
    assert device._process is device._connection is device._monitor is None
    assert device.stats["error"] != 0
    device.close()  # Cleanup is terminal and idempotent, even after reporting failure.
    gc.collect()
    assert not directory.exists()


@pytest.mark.parametrize("failure", ["fail_cancel", "fail_close", "exit_close"])
def test_process_reset_does_not_reconnect_after_shutdown_failure(
    process_factory, tmp_path, failure
):
    log = tmp_path / "lifecycle"
    with devices.Device(
        process_factory, config={failure: True, "lifecycle_log": str(log)}
    ) as device:
        device.start()
        assert device.source.reset() == n.StreamStatus.SOURCE_FAILURE
        assert log.read_text().splitlines() == ["open", "cancel", "close"]
        assert device._process is device._connection is device._monitor is None
        assert device.stats["error"] != 0
