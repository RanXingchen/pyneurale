# Offline typed kinematics

## Derive movement measures

```python
import numpy as np
from neurale.data import SignalArray
from neurale.features import speed, velocity

time = np.arange(11) / 10
position = SignalArray.from_array(
    np.column_stack((2 * time, -3 * time)), fs=10, time=time,
    channel_names=("x", "y"), channel_types="behavior",
    units="m", name="position",
)
movement_velocity = velocity(position)
movement_speed = speed(position)
print(movement_velocity.data.shape, movement_speed.data.shape)
```

Both results retain the source sample times. `data` uses
`(trial, sample, spatial_dimension)` order, with a single dimension for speed.

`neurale.features.kinematics` provides offline transforms for typed position,
trajectory, and velocity signals. Decoder targets, model fitting, experiment
state, and online processors are outside this module.

## Representation and axes

`KinematicSeries.data` always uses `(trial, sample, spatial_dimension)` order.
Its explicit time array uses `(trial, sample)`. A `SignalArray` becomes one
ungrouped trial; `kinematic_trials()` obtains trial epochs through the shared
`split_recording_trials()` implementation and requires equal sample counts
for the rectangular trial axis. Clock, nominal sampling rate, absolute/local
timestamps, channel-backed dimension names, units, source signal attrs, complete
typed `Trial` values, and trial IDs are retained.

The nominal sampling rate remains metadata even when timestamps are irregular.
Numerical derivatives always use the explicit timestamps and therefore do not
replace irregular timing with `1 / sampling_rate`.

## Numerical contract

The first implementation supports:

- `kinematic_derivative()` for position/trajectory orders 1 and 2, and
  velocity order 1;
- `velocity()` from position or trajectory;
- `acceleration()` from position, trajectory, or velocity;
- Euclidean `speed()`;
- planar `direction()` for exactly two spatial dimensions.

The only derivative method is `gradient`: centered finite differences at
interior samples and one-sided differences at both boundaries. First
derivatives use first-order boundary estimates and require at least two samples
per trial. Second derivatives use repeated gradients with second-order boundary
estimates and require at least three samples per trial. Other quantity/order
combinations are rejected because the current quantity contract has no `jerk`
or generic derivative representation. Results retain every input sample
timestamp; no padding or sample deletion is performed. No smoothing is implicit,
and the only accepted smoothing setting is `none`.

NaNs are rejected by default. `nan_policy="propagate"` preserves NumPy's local
finite-difference propagation. Infinite inputs are always rejected. Speed and
direction require identical component units. Speed uses a scaled hypot
reduction so finite very large and subnormal vectors are not corrupted by a
naive square-sum. A true norm outside the finite float64 range raises
`ValidationError` without leaking a floating-point warning. Direction is
`atan2(v_y, v_x)` mapped into `[0, 2*pi)` radians; stationary samples have no
defined direction and return NaN.
