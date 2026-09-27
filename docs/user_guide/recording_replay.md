# Record and inspect a session

Follow [source installation](../getting_started/index.md) and select the
`nrf` extra before building the project.

## Record a closed-loop run

Create the recorder for the device before constructing a session, and pass it as
`recording=`. The session attaches the recorder, stops acquisition, drains the
trace, and finalizes the `.nrf` file. Do not close it independently while the
session is running.

```python
from neurale.recording import RecorderConfig, SessionRecorder

recorder = SessionRecorder.create(RecorderConfig(path="session.nrf"), device)
# Pass recording=recorder to CenterOutSession or SSVEPSession.
```

`device` is the acquisition device used by the session. The
[Center-Out example](center_out.md) accepts `--output` for a complete runnable
recording path. After `session.run()`, check `outcome.recording.complete` before
using the file. A missing or incomplete result is not a valid finished session.

The default recorder limits are derived from nominal acquisition cadence. Use
explicit `RecorderLimits` for faster-than-real-time replay or variable framing.
Disk throughput must keep up with acquisition; a full queue faults recording
instead of dropping data silently. A spool may remain after failure. Do not
repair or finalize it while recording I/O is still active. See the
[recording API](../api/python.md) for limits and the
[recovery design](../development/native_recording_replay.md) for failure handling.

## Read the finished file

```python
from neurale.io.nrf import NrfReader

with NrfReader.open("session.nrf") as reader:
    print(reader.manifest["streams"])
    neural = reader.read_stream("neural", 0, 1000)
```

The stream name must exist in that session's manifest; `"neural"` is the name
used by the supplied Center-Out example. `read_stream` selects a half-open
sample range and does not require extracting the entire package.

## Build a native replay image

NRF reading and native frame replay serve different purposes. To prepare
recorded frames for the native replay source, choose an explicit topology and
build a separate image:

```python
from neurale.recording import ReplayConfig, build_replay_image

with build_replay_image(
    "session.nrf",
    ReplayConfig(mode="exact_frames"),
    path="session.replay",
) as replay:
    print(replay.metadata)
```

`exact_frames` retains recorded frame grouping; other modes intentionally change
selection or framing. The image is not a replacement for the NRF session. The
builder verifies the source by default and leaves it unchanged. Native replay
execution currently belongs to the C++ source and has no Python binding; this
Python example prepares and inspects the image, but does not run the source.
See the [recording design](../development/native_recording_replay.md) for
selection, pacing, and injected-fault contracts.
