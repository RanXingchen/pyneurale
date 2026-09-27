# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from neurale import devices, streaming
from neurale.devices import lsl


def test_import_is_lazy_and_missing_dependency_is_actionable():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from neurale import devices
from neurale.devices import lsl
assert "pylsl" not in sys.modules
assert "lsl" in devices.list()
sys.modules["pylsl"] = None
try:
    lsl.discover()
except Exception as error:
    assert "pyneurale[lsl]" in str(error) and "PYLSL_LIB" in str(error)
else:
    raise AssertionError("missing dependency accepted")
""",
        ],
        check=True,
    )


class Info:
    def __init__(self, fmt=1, rate=1000, source="test", units=("uV", "")):
        self.fmt, self.rate, self.source, self.units = fmt, rate, source, units

    def name(self):
        return "test"

    def type(self):
        return "EEG"

    def source_id(self):
        return self.source

    def uid(self):
        return self.source + "-uid"

    def hostname(self):
        return "localhost"

    def channel_count(self):
        return 2

    def nominal_srate(self):
        return self.rate

    def channel_format(self):
        return self.fmt

    def as_xml(self):
        return (
            "<info><desc><channels>"
            + "".join(f"<channel><unit>{unit}</unit></channel>" for unit in self.units)
            + "</channels></desc></info>"
        )


class Inlet:
    def __init__(self, info, **kwargs):
        self.description, self.options = info, kwargs
        self.chunks = []
        self.closed = False
        self.reset = False

    def info(self, timeout):
        return self.description

    def open_stream(self, timeout):
        pass

    def close_stream(self):
        self.closed = True

    def flush(self):
        pass

    def was_clock_reset(self):
        return self.reset

    def pull_chunk(self, *, timeout, max_samples, dest_obj):
        if self.chunks:
            chunk = self.chunks.pop(0)
            if isinstance(chunk, Exception):
                raise chunk
            values, timestamps = chunk
            dest_obj[: len(timestamps)] = values
            return None, timestamps
        return None, []


@pytest.fixture
def fake(monkeypatch):
    info = Info()
    module = SimpleNamespace(
        resolve_streams=lambda wait_time: [info], StreamInlet=Inlet, local_clock=lambda: 10.0
    )
    monkeypatch.setattr(lsl, "_lsl", lambda: module)
    monkeypatch.setattr(lsl, "_correction", lambda inlet, timeout: (2.0, 8.0, 0.001))
    monkeypatch.setattr(lsl, "host_time_ns", lambda: 20_000_000_000)
    return info, module


def test_discovery_and_full_description(fake):
    assert lsl.discover()[0]["source_id"] == "test"
    result = lsl.inspect(source_id="test")
    assert result["channels"] == [{"label": "ch1", "unit": "uV"}, {"label": "ch2", "unit": ""}]
    assert "<info>" in result["xml"]


@pytest.mark.parametrize("fmt,dtype", [(1, "float32"), (2, "float64"), (4, "int32"), (5, "int16")])
def test_signal_contract(fake, fmt, dtype):
    fake[0].fmt = fmt
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    signals = p.describe()
    assert [s.name for s in signals] == ["data", "timestamps"]
    assert signals[0].dtype == dtype and signals[0].unit == "unspecified"
    assert signals[0].channel_names == ("ch1", "ch2")
    assert signals[1].channels == 2
    assert p._inlet.options["recover"] is False
    p.close()


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_samples", 0),
        ("buffer_seconds", 0.5),
        ("data_timeout", float("nan")),
        ("max_lag", -1),
        ("unknown", 1),
    ],
)
def test_invalid_options(fake, option, value):
    with pytest.raises(ValueError):
        lsl.LSLProvider().open({"source_id": "test", option: value})


def test_resolution_errors(fake):
    with pytest.raises(ValueError, match="selector"):
        lsl.LSLProvider().open({})
    with pytest.raises(TimeoutError):
        lsl.LSLProvider().open({"source_id": "missing"})
    fake[1].resolve_streams = lambda wait_time: [Info(), Info()]
    with pytest.raises(ValueError, match="ambiguous"):
        lsl.LSLProvider().open({"stream_type": "EEG"})


@pytest.mark.parametrize("fmt,rate", [(3, 1000), (6, 1000), (7, 1000), (1, 0)])
def test_unsupported_stream(fake, fmt, rate):
    fake[0].fmt, fake[0].rate = fmt, rate
    p = lsl.LSLProvider()
    try:
        with pytest.raises(ValueError):
            p.open({"source_id": "test"})
    finally:
        p.close()


class Ingress:
    def __init__(self, stop_after=1):
        self.frames = []
        self.cancelled = False
        self.failed = False
        self.stop_after = stop_after

    def publish_frame(self, blocks):
        self.frames.append([(index, data.copy(), dict(meta)) for index, data, meta in blocks])
        self.cancelled = len(self.frames) >= self.stop_after

    def fail(self):
        self.failed = True


def test_sample_timestamps_mapping_and_pairing(fake):
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    p._inlet.chunks = [(np.array([[1, 2], [3, 4]]), [7.998, 7.999]), (np.array([[5, 6]]), [8.0])]
    ingress = Ingress(2)
    p._acquire(ingress)
    first, second = ingress.frames
    np.testing.assert_array_equal(first[0][1], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(first[1][1][:, 0], [7.998, 7.999])
    np.testing.assert_allclose(first[1][1][:, 1], [19.998, 19.999], rtol=0, atol=1e-12)
    assert first[0][2] == first[1][2]
    assert first[0][2]["uncertainty_ns"] == 1_000_000
    assert second[0][2]["sample_index"] == second[1][2]["sample_index"] == 2
    assert second[0][2]["device_tick"] == 2_000_000
    p.close()


@pytest.mark.parametrize("times", [[8.0, 7.999], [float("nan")], [7.0], [9.0]])
def test_timestamp_faults(fake, times):
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    p._inlet.chunks = [(np.zeros((len(times), 2)), times)]
    ingress = Ingress()
    p._run(ingress)
    assert ingress.failed and not ingress.frames
    with pytest.raises(RuntimeError, match="LSL acquisition failed"):
        p.close()
    assert p._inlet is None


@pytest.mark.parametrize("fault", ["reset", "read", "pause", "stale"])
def test_lifecycle_faults(fake, monkeypatch, fault):
    p = lsl.LSLProvider()
    p.open({"source_id": "test", "data_timeout": 0.002})
    p._sync(1)
    if fault == "reset":
        p._inlet.reset = True
    if fault == "read":
        p._inlet.chunks = [RuntimeError("read failed")]
    if fault == "stale":
        p._synced_at -= 6
    ingress = Ingress()
    p._run(ingress)
    assert ingress.failed
    with pytest.raises(RuntimeError):
        p.close()


def test_cancellation_and_repeated_close(fake):
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p.start(Ingress())
    p.cancel()
    p.close()
    p.close()
    assert not p._thread.is_alive()


def test_sync_refresh_timeout_expires(fake, monkeypatch):
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    base = p._synced_at
    ticks = iter([base, base + 2, base + 6])
    monkeypatch.setattr(lsl.time, "monotonic", lambda: next(ticks))

    def timeout(*args):
        raise TimeoutError("no mapping")

    monkeypatch.setattr(lsl, "_correction", timeout)
    ingress = Ingress()
    p._run(ingress)
    assert ingress.failed
    with pytest.raises(RuntimeError, match="stale"):
        p.close()


def test_cached_correction_expires_while_samples_continue(fake, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(lsl.time, "monotonic", lambda: clock[0])
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    ingress = Ingress(stop_after=100)

    def pull(**kwargs):
        clock[0] += 1.0
        kwargs["dest_obj"][:1] = 1
        return None, [8.0]

    p._inlet.pull_chunk = pull
    p._run(ingress)
    assert ingress.frames  # The data path remains active, only clock probing stalls.
    assert ingress.failed
    assert p._synced_at == 100.0
    with pytest.raises(RuntimeError, match="stale"):
        p.close()


def test_only_new_measurements_refresh_sync(fake, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(lsl.time, "monotonic", lambda: clock[0])
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    initial_mapping = p._mapping
    clock[0] += 1.1
    p._sync(0)
    assert p._synced_at == 100.0 and p._mapping == initial_mapping
    # A newly observed measurement is itself 0.5 seconds old.
    fake[1].local_clock = lambda: 11.5
    monkeypatch.setattr(lsl, "_correction", lambda inlet, timeout: (2.0, 9.0, 0.001))
    p._sync(0)
    assert p._synced_at == pytest.approx(100.6)
    p.close()


def test_initial_cached_measurement_keeps_its_age(fake, monkeypatch):
    monkeypatch.setattr(lsl.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(lsl, "_correction", lambda inlet, timeout: (2.0, 2.0, 0.001))
    p = lsl.LSLProvider()
    p.open({"source_id": "test"})
    p._sync(1)
    ingress = Ingress()
    p._run(ingress)
    assert ingress.failed and not ingress.frames
    with pytest.raises(RuntimeError, match="stale"):
        p.close()


@pytest.mark.parametrize("destroy", [False, True])
def test_live_pause_or_destroy_reports_failure(destroy):
    pylsl = pytest.importorskip("pylsl")
    identifier = "neurale-failure-" + uuid.uuid4().hex
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo("fault", "EEG", 2, 1000, "float32", identifier))
    device = devices.open("lsl", source_id=identifier, data_timeout=0.1, timeout=10)
    try:
        device.start()
        if destroy:
            del outlet
        deadline = time.monotonic() + 3
        while device.stats["error"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert device.stats["error"] != 0
        with pytest.raises(Exception, match="LSL acquisition failed"):
            device.close()
    finally:
        device.close()


@pytest.mark.parametrize("recreate", [False, True])
def test_live_outlet_spawn_and_runner(recreate):
    pylsl = pytest.importorskip("pylsl")
    identifier = "neurale-test-" + uuid.uuid4().hex
    info = pylsl.StreamInfo("test", "EEG", 2, 1000, "float32", identifier)
    outlet = pylsl.StreamOutlet(info)
    frames = []

    class Capture:
        def write(self, frame):
            frames.append(frame)

        def handle_discontinuity(self, gap):
            pytest.fail("unexpected gap")

    device = devices.open("lsl", source_id=identifier, max_samples=32, timeout=10)
    runner = streaming.StreamRunner(
        device.schema,
        streaming.RealtimeConfig(),
        device.source,
        streaming.IdentityNativeProcessor(),
        streaming.PythonSinkAdapter(Capture()),
        profile="research",
    )
    worker = None
    try:
        for repetition in range(2):
            if repetition:
                if recreate:
                    del outlet
                    outlet = pylsl.StreamOutlet(info)
                assert runner.reset() == streaming.StreamStatus.OK
            assert runner.prepare() == streaming.StreamStatus.OK
            assert runner.arm() == streaming.StreamStatus.OK
            device.start()
            frames.clear()
            worker = threading.Thread(target=runner.run)
            worker.start()
            values = np.arange(24, dtype=np.float32).reshape(12, 2)
            stamps = pylsl.local_clock() - 0.020 + np.arange(12) * 0.001
            outlet.push_chunk(values, stamps.tolist())
            deadline = time.monotonic() + 3
            while sum(f.blocks[0].n_samples for f in frames) < 12 and time.monotonic() < deadline:
                time.sleep(0.005)
            runner.stop()
            worker.join(3)
            assert not worker.is_alive()
            data, timestamps = [], []
            for frame in frames:
                assert len(frame.blocks) == 2
                a, b = frame.blocks
                assert a.n_samples == b.n_samples and a.sample_idx_start == b.sample_idx_start
                data.append(
                    np.frombuffer(
                        frame.payload,
                        dtype=np.float32,
                        count=a.n_samples * 2,
                        offset=a.payload_offset,
                    ).reshape(-1, 2)
                )
                timestamps.append(
                    np.frombuffer(
                        frame.payload,
                        dtype=np.float64,
                        count=b.n_samples * 2,
                        offset=b.payload_offset,
                    ).reshape(-1, 2)
                )
            np.testing.assert_array_equal(np.concatenate(data), values)
            actual = np.concatenate(timestamps)
            np.testing.assert_array_equal(actual[:, 0], stamps)
            assert np.all(np.diff(actual[:, 1]) > 0)
            assert abs(lsl.host_time_ns() * 1e-9 - actual[-1, 1]) < 1
            assert device.stats["error"] == 0
    finally:
        runner.stop()
        if worker is not None:
            worker.join(3)
        device.close()
