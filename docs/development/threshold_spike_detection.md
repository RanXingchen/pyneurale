# Threshold spike detection reference contract

`neurale.sorting.DetectionConfig` is the public immutable configuration for
threshold detection. A private Python reference oracle defines the numerical
contract for small inputs and parity tests. The public offline production entry
point is `neurale.sorting.detect_threshold_spikes`; it delegates to the native
CPU kernel and returns one `SpikeWaveformBatch` per continuous segment. There is
no public online detector; fixed-capacity online detection remains private
native pipeline machinery.

## Input and noise estimate

The oracle accepts one regularly sampled, continuous `SignalArray` segment in
sample-major `(sample, channel)` order. It does not filter, rereference, infer
discontinuities, or convert the waveform dtype. The caller supplies the segment
identity and, when the signal is a slice of a larger stream, its absolute sample
index offset.

For channel $c$:

```text
m[c]     = median(x[:, c])
noise[c] = median(abs(x[:, c] - m[c])) / 0.6744897501960817
T[c]     = threshold_multiplier * noise[c]
```

This centered MAD definition is intentionally location invariant. A channel
with zero MAD has no eligible crossings.

Location invariance holds while the integer waveform is exactly representable
as `float64`: the geometry is computed in `float64`, and past the 2^53 precise
integer limit small oscillations collapse onto the DC offset, which would
silently flip a detection to an empty batch. Integer inputs that do not
round-trip exactly to `float64` are therefore rejected with `ValidationError`
rather than producing a wrong result, keeping the Python oracle aligned with
the native double-precision kernel.

## Crossing, refractory suppression, and alignment

A candidate begins at the first sample of an inclusive threshold excursion:

```text
negative: x[i, c] - m[c] <= -T[c]
positive: x[i, c] - m[c] >=  T[c]
both:     abs(x[i, c] - m[c]) >= T[c]
```

Without explicit electrode groups, every input channel is independent. An
explicit group list must partition every zero-based `ChannelTable` position
exactly once. Group order defines the integer group identifier stored in the
result.

Crossings at the same sample in one group are collapsed to the largest
normalized threshold excursion, with the lower channel position breaking a
tie. Remaining crossings are ordered by crossing sample, and the earliest
crossing wins. A later crossing is accepted when:

```text
later_sample - previous_accepted_sample
    >= ceil(refractory_interval * sampling_rate)
```

Different electrode groups apply refractory suppression independently.

After refractory suppression, the detector searches the closed sample interval
centered on each accepted crossing with the configured
`alignment_search_radius`. Alignment selects the earliest minimum, maximum, or
absolute maximum for negative, positive, or both polarity respectively. The
search interval is clipped to available input; waveform boundaries are handled
separately. This ordering is normative: refractory suppression uses crossing
samples, not aligned peak samples.

## Native CPU contract

The CPU kernel consumes exact `float64`, C-contiguous, sample-major
`(sample, channel)` data. Its binding rejects other dtypes and strided layouts
instead of silently converting or transposing them. Noise estimation copies one
strided channel at a time into one reusable `n_samples` scratch buffer; the
threshold scan itself walks sample-major input. Candidate storage is bounded by
the number of input sample/channel positions, and sorting is confined to each
electrode group's crossing list rather than using a quadratic pair structure.

The private native event result contains aligned sample index, original crossing
index, peak channel, group identifier, centered amplitude, normalized score,
polarity, and the per-channel center/noise/threshold arrays. A shared
multi-chunk native path extracts event-major waveforms after the final event
order is known; the event-only binding remains available privately.

## Whole-segment and chunked processing

`detect_threshold_spikes(signal, config, chunk_size=None)` supplies each
continuous segment as one native chunk. A positive `chunk_size` supplies the
same segment as consecutive logical chunks. Both modes compute centered MAD
over the complete continuous segment and share crossing, refractory, alignment,
and waveform logic. Each source sample is scanned once in order, active state
continues across chunks, and waveform access may read neighboring chunks.
Consequently, logical chunk edges neither duplicate events nor behave like
waveform boundaries.

The public return type is always `tuple[SpikeWaveformBatch, ...]`, with one
batch per retained continuous segment. Without discontinuities, a non-empty or
empty signal produces a one-element tuple. This fixed return shape avoids a
single batch spanning segment IDs and avoids changing return type only when a
gap happens to be present.

`sample_index_offset` identifies the absolute source position of the first
input row. Segment-local native indices are translated back through their
source segment start before this offset is added, so chunking and gap removal
do not renumber source samples.

Instantaneous `EventSeries` discontinuities are state barriers at their onset
without discarding a source sample. Duration discontinuities exclude samples
in their half-open `[onset, stop)` interval. Discontinuity times use the shared
clock conversion contract. Detection, refractory state, alignment windows, and
waveforms never cross a segment boundary.

## Waveforms and boundaries

The fixed waveform layout is `(spike, sample, channel)` with length:

```text
pre_samples + 1 + post_samples
```

The aligned sample is at `waveforms[:, pre_samples, :]`. `drop` discards a
candidate whose fixed waveform would cross an input boundary. `raise` reports a
`ValidationError`. Padding and clipping are not supported because they would
fabricate samples or violate the fixed typed waveform contract.

The oracle and public native path return `SpikeWaveformBatch`, preserving the selected signal's
channel metadata, sampling rate, clock, absolute times, source name, segment
identity, and waveform dtype. It performs no detection across segments.

## Output amplitudes and polarity

`amplitudes` are baseline-centered detection excursions at the aligned peak
channel -- the same centered signal used for thresholding and polarity, not the
raw source sample. Their sign therefore matches `spike_polarities`, and they are
invariant to a DC offset, consistent with the centered-MAD geometry. With a
non-zero baseline, a negative spike can have a positive raw value; `amplitudes`
records the centered excursion (negative) while the raw value stays positive.

`waveforms` keep the source dtype and the un-centered signal, so the raw aligned
value is still recoverable as:

```text
waveforms[np.arange(n_spikes), pre_samples, peak_channel_indices]
```

Each spike may peak on a different channel, so the per-spike channel selector
``peak_channel_indices`` must be indexed sample-by-sample rather than with a
single shared ``peak_channel``.

`spike_polarities` is populated only for `polarity="both"`; the per-spike sign is
`-1` for a negative excursion and `+1` for a positive one, derived from the
centered detection signal.
