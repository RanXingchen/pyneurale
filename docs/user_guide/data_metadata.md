# Data metadata value contract

PyNeurale fields described as **deep-frozen metadata** accept a closed value
domain. This applies to fields such as `SignalArray.attrs`, `ChannelInfo.attrs`,
`Clock.attrs`, `Event.attrs`, `EventSeries.attrs`, `Trial.attrs`, `Frame.attrs`,
`TrialEpoch.recording_metadata`, `TrialEpoch.trial_attrs`,
`SpikeWaveformBatch.attrs`, and the deep-frozen provenance mappings in typed
ERP, kinematics, and tuning results.

The closed domain prevents an immutable typed object from containing a mutable
leaf that can be changed through an alias.

## Supported values

The following values are accepted recursively:

- `None`;
- Python `bool`, `int`, `float`, `str`, and `bytes` values;
- NumPy scalar values;
- mappings;
- lists and tuples;
- sets and frozensets;
- NumPy ndarrays;
- the explicitly admitted immutable PyNeurale values `ChannelInfo`,
  `ChannelTable`, and `Clock`.

Mappings become immutable mappings. Lists and tuples become tuples. Sets and
frozensets become frozensets. Container values are recursively checked against
this same domain.

A non-object ndarray is copied into immutable bytes-backed storage. Its dtype
and shape are retained, and its writeable flag cannot subsequently be enabled.
An object-dtype ndarray is instead converted with `tolist()` and recursively
frozen; it therefore becomes a tuple or nested tuple rather than remaining an
ndarray. Every converted element must itself be supported.

`ChannelInfo`, `ChannelTable`, and `Clock` are copied as immutable metadata
leaves. Their constructors validate their fields first. In particular,
`ChannelInfo.impedance` is either `None` or a finite real/complex scalar; arrays
and mutable containers are rejected.

## Unsupported values

Unsupported leaves raise `neurale.exceptions.ValidationError`. Examples include
`bytearray`, Python `datetime`, `UUID`, `pathlib.Path`, enum members, arbitrary
user dataclasses (including `frozen=True` dataclasses), and custom objects.
Freezing a dataclass is shallow and does not prove that values referenced by its
fields are immutable, so user dataclasses are not admitted implicitly.

The metadata freezer does not impose field-specific numerical rules on the
supported scalar domain. Individual typed objects may impose additional rules,
such as finite timestamps or finite channel impedance.

## Scope

Only fields documented as deep-frozen use this closed domain. Some mutable
aggregate objects retain explicit copy semantics instead. For example,
`Recording.metadata`, `FeatureMatrix.attrs`, `SpikeTrain.attrs`,
`TrialTable.attrs`, and `ElectrodeArray.attrs` are currently copied mutable
application mappings; the deep-frozen contract must not be inferred for those
fields.
