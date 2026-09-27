# Python API

Module pages are generated from Python docstrings by Sphinx Autosummary.

```{eval-rst}
.. autosummary::
   :toctree: _generated

   neurale
   neurale.exceptions
   neurale.io
   neurale.io.nrf
   neurale.recording
   neurale.pipeline
   neurale.streaming
   neurale.data
   neurale.data.alignment
   neurale.runtime
   neurale.runtime.capabilities
   neurale.runtime.config
   neurale.runtime.context
   neurale.runtime.dependencies
   neurale.runtime.diagnostics
   neurale.runtime.initialization
   neurale.runtime.random
   neurale.runtime.threading
   neurale.signal
   neurale.signal.simulation
   neurale.devices
   neurale.devices.provider
   neurale.devices.lsl
   neurale.devices.simulation
   neurale.experiments
   neurale.experiments.assistance
   neurale.experiments.center_out
   neurale.experiments.speech
   neurale.experiments.ssvep
   neurale.experiments.traceability
   neurale.experiments.webgrid
   neurale.features
   neurale.features.bandpower
   neurale.features.erp
   neurale.features.kinematics
   neurale.features.oscillations
   neurale.features.preprocessing
   neurale.features.tuning
   neurale.decoding.kalman
   neurale.decoding.lda
   neurale.decoding.linear
   neurale.decoding.persistence
   neurale.decoding.sequence
   neurale.decoding.torch
   neurale.models
   neurale.models.evaluation
   neurale.models.linear_model
   neurale.models.manifold
   neurale.models.preprocessing
   neurale.models.state_space
   neurale.sorting
   neurale.sorting.clustering
   neurale.sorting.curation
   neurale.sorting.metrics
   neurale.sorting.workflow
```

The [provider guide](../user_guide/device_providers.md) covers independently
installed adapters; the [LSL guide](../user_guide/lsl.md) covers the optional
continuous-stream provider. `neurale.devices.provider` documents the wheel-shipped
Python SDK and C header lookup, while `neurale.devices.lsl` documents discovery
and stream inspection.

The optional presentation reference documents Python wrappers from docstrings
and lists native-only symbols without importing the optional extension.

```{toctree}
:maxdepth: 1

presentation
```

## Runtime logging

Logging is part of the public runtime API. Its implementation module remains
private; import these functions directly from `neurale.runtime`.

```{eval-rst}
.. currentmodule:: neurale.runtime

.. autosummary::

   configure_logging
   is_logging_configured
   reset_logging
```

## Interoperability I/O

`neurale.io` provides focused MAT and CSV interoperability. Importing the
module does not import SciPy; format-specific dependencies are resolved only
when an I/O operation is called. The implementation modules remain private;
import these functions directly from `neurale.io`.

```{eval-rst}
.. currentmodule:: neurale.io

.. autosummary::

   loadmat
   savemat
   loadcsv
   savecsv
```

## NRF v1 recording sessions

`neurale.io.nrf` reads, writes, diagnoses, and repairs offline NRF v1 sessions
according to the [format specification](https://github.com/RanXingchen/pyneurale/blob/main/specifications/nrf/v1/README.md).
Install the `nrf` extra for its storage and validation dependencies; importing
the module alone does not load them.

```{eval-rst}
.. currentmodule:: neurale.io.nrf

.. autosummary::

   NrfWriter
   NrfReader
   ManifestBuilder
   record_field
   write_recording
   diagnose_session
   recover
   read_report
   SessionDiagnosis
   Diagnostic
   RecoveryResult
   IgnoredRecord
```

`diagnose_session` and `recover` are also reachable as a command line:
`python -m neurale.io.nrf diagnose <session>.nrf` reports without mutating,
and `python -m neurale.io.nrf recover <session>.nrf` rebuilds caches and
writes a recovery report.

## Streaming and recording

`neurale.streaming` controls the native runtime; `neurale.recording` connects a
live stream to NRF storage. `SessionRecorder` attaches to a lossless runtime
edge, writes accepted messages to a disk spool, and finalizes one compressed
`.nrf` file after capture stops. There is no Python recording fallback.

`RecorderConfig.spool_retention` defaults to
`"delete_after_validated_finalization"`; use `"retain"` to keep a successful
spool. `SessionRecorder.spool_path` identifies a retained spool after failure,
and `RecorderStatus.cleanup_paths` identifies residual cleanup targets.
`SessionRecorder.finalization_progress` supports non-realtime progress polling.
Session files are read through indexed chunks without extraction.

Omitted `RecorderLimits` budget five seconds of acquisition backlog plus 25%
margin. `spool_capacity_bytes` limits disk use, not locked RAM. Sustained disk
throughput must exceed input throughput; queue overflow or disk errors fault
recording. Do not finalize or repair a live spool, and close the recorder
explicitly before using its output. Buffered durability does not guarantee
survival of a power loss. See the [recording design](../development/native_recording_replay.md)
for recovery and durability details.

For a prepared single-signal device, recording intent is deliberately small:

```python
config = RecorderConfig(path="session.nrf")
recorder = SessionRecorder.create(config, device)
```

The recorder derives signal metadata from the prepared device. For a
multi-signal source, select a `SignalSchema`, not a bare integer ID. Session
identity and writer provenance are recorder-owned; descriptive metadata and
resource bounds can be supplied separately:

```python
from neurale.recording import RecorderLimits, RecorderSessionMetadata

config = RecorderConfig(
    path="session.nrf",
    session=RecorderSessionMetadata(
        subject={"id": "participant-01"},
        experiment={"name": "center-out"},
    ),
    limits=RecorderLimits(
        spool_capacity_bytes=128 << 20,
        max_control_records=32_000,
    ),
)
```

`max_control_records` bounds control records. Critical recording is
lossless-until-fault; there is no lossy overflow-policy option.

```{eval-rst}
.. currentmodule:: neurale.recording

.. autosummary::

   SessionRecorder
   RecorderConfig
   RecorderLimits
   RecorderSessionMetadata
   RecorderStatus
   RecorderState
   StreamRecording
   StreamMetadata
   StreamStorageConfig
   StreamTimingConfig
   StreamTimingMode
   StreamProvenanceConfig
   StreamRole
   block_index_columns
   read_block_index
```

## Decoding contract

Classical decoders consume prepared `FeatureMatrix` frames and aligned targets.
They return a `SignalArray` for continuous targets or a
`ClassificationPrediction` for classes. Decoders verify alignment, feature
identity, and clock domain; the caller performs feature extraction, alignment,
and any feature selection beforehand. A fitted `FeatureSchema` fingerprint
rejects incompatible prediction inputs.

```{eval-rst}
.. currentmodule:: neurale.decoding

.. autosummary::

   FeatureSchema
   ContinuousTargetSchema
   ClassificationTarget
   ClassificationPrediction
   FitSegment
   BaseDecoder
   ContinuousDecoder
   ClassificationDecoder
```

## Kalman decoding

`KalmanDecoder` uses a `LinearGaussianStateSpace` model and a `KalmanFilter`,
with an optional frozen feature scaler. `fit_segments` accepts independent
segments without creating transitions across their boundaries; gaps inside a
segment are rejected. The fitted parameters and scaler cannot be refitted
through the decoder. Prediction is stateful and chunk-consistent; `reset()`
restores the fitted initial state or accepts a caller-supplied state and
covariance. Feature subsets are selected by caller-supplied names and order.

```{eval-rst}
.. currentmodule:: neurale.decoding

.. autosummary::

   KalmanDecoder
```

## Regression and discriminant decoding

`LinearDecoder` and `RidgeDecoder` fit continuous targets with multi-output
least squares and L2-penalized regression. `LDADecoder` predicts classes and
probabilities; probability column `i` corresponds to `classes_[i]`. The
predicted label follows the model's decision rule, including probability ties.

All three accept an optional frozen scaler and `TemporalContext`, applied in
that order. Context removes boundary frames from targets and predicted
timestamps as well as features. Prediction is stateless, and a rejected refit
leaves the previous fit usable. Feature ranking, PCA, probability smoothing,
and decision thresholds are separate caller-owned steps.

```{eval-rst}
.. currentmodule:: neurale.decoding

.. autosummary::

   LinearDecoder
   RidgeDecoder
   LDADecoder
```

## Sequence decoding

`neurale.decoding.sequence` turns per-step token scores into token sequences;
it does not consume feature matrices or implement `BaseDecoder`. `Vocabulary`
sets score-column order and tie-breaking order. `eos` has a scored column;
`bos` is context only.

`NgramLanguageModel` exposes unigram/bigram counts and probabilities. Add-k
smoothing defaults to off, so unseen events have zero probability. A fitted
model retains its vocabulary across rejected refits. `BeamSearchDecoder`
returns bounded `Hypothesis` results with separate model and language scores.
Chunked and batch decoding agree. `reset()` adopts a snapshot of the fitted
language model; a refit during one sequence does not change that sequence's
prior.

```{eval-rst}
.. currentmodule:: neurale.decoding.sequence

.. autosummary::

   Vocabulary
   NgramLanguageModel
   BeamSearchDecoder
   Hypothesis
```

## Decoder persistence

`neurale.decoding.persistence` stores one fitted decoder in a versioned
directory with a canonical JSON manifest and `.npy` arrays. Loading never
unpickles arrays or follows an import path from the artifact. It validates the
decoder type, version, fitted device, feature schema, model dimensions, and
stored arrays before reconstructing the decoder. A loaded decoder preserves
the saved prediction behavior and fitted device.

Online prediction state is omitted by default, so a loaded decoder starts
post-reset. Use `runtime_state=True` to resume a running decoder, including a
sequence decoder's current language-model snapshot. Saves are published
atomically where supported by the platform.

`DecoderArtifactVersionError` indicates an unsupported format or decoder
version; `DecoderArtifactFormatError` indicates a malformed manifest;
`DecoderArtifactCorruptionError` indicates missing or invalid arrays. All
derive from `DecoderArtifactError` and `ValueError`. `read_manifest` parses
metadata without validating whether the decoder can be loaded.

```{eval-rst}
.. currentmodule:: neurale.decoding.persistence

.. autosummary::

   save_decoder
   load_decoder
   read_manifest
   supported_types
   DecoderArtifactError
   DecoderArtifactVersionError
   DecoderArtifactFormatError
   DecoderArtifactCorruptionError
```

## The optional Torch boundary

`neurale.decoding.torch` provides optional-dependency helpers, not a
Torch-backed decoder. Importing this module or `neurale.decoding` does not load
Torch. Calling a helper that requires an unavailable Torch installation raises
{class}`~neurale.exceptions.DependencyError`. There is no `torch` extra.

```{eval-rst}
.. currentmodule:: neurale.decoding.torch

.. autosummary::

   torch_status
   has_torch
   require_torch
```
