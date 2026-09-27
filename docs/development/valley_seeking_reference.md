# Valley Seeking reference contract

The private, deterministic CPU reference implements the fixed-neighborhood
Valley Seeking rule described by Koontz and Fukunaga in
[*A Nonparametric Valley-Seeking Technique for Cluster Analysis*](https://www.ijcai.org/Proceedings/71/Papers/037.pdf).
It is a small-input numerical oracle for a later optimized implementation, not
a public clustering API. It does not expose a monolithic sorter hierarchy.

## Input and neighborhood

`features` has exact `float64` shape `(n_observations, n_features)`, with at
least one feature column and only finite values. `initial_labels` is an exact
`int64` vector with one value per observation. Labels are nominal values; they
need not be contiguous or non-negative. Their deterministic order is ascending
numeric order.

The positive finite `radius` is the bandwidth of a uniform Euclidean ball.
Observation `j` is a neighbor of `i` exactly when `i != j` and:

```text
euclidean_distance(features[i], features[j]) <= radius
```

The relation is symmetric and fixed before iteration. Duplicate observations
are neighbors; an observation is never its own neighbor. The inclusive radius
boundary is exact. `neighbor_counts` is an
unnormalized, uniform-ball density proxy only. The reference does not estimate
a KDE, weight votes by distance, choose a bandwidth, or transform the input.

Squared normalized distance is accumulated in feature order with one fused
multiply-add per feature and stops as soon as the sum exceeds one. The native
kernel uses `std::fma`. The Python reference uses `math.fma` where available;
on older Python versions it computes the exact rational multiply-add and rounds
once to binary64. It deliberately does not use `numpy.dot`, so a BLAS/provider
change cannot alter membership for a pair at the radius boundary.

## Vote, ties, labels, and termination

Each iteration counts the current labels in every fixed neighborhood and then
updates all observations synchronously. Thus no observation can see a label
written earlier in the same iteration.

The label with most neighbor votes wins. If the current label is tied for the
maximum, it is retained, following the mathematical source. If a tie does not
include the current label, the smallest label in the ascending label order
wins. An isolated observation has zero votes for every label, so its current
label is retained.

The set of possible output labels is the set present in `initial_labels`.
Iteration stops after a complete pass changes no labels, or after the positive
`max_iterations` bound. The result reports `converged`, `n_iterations`, and one
of `empty`, `converged`, or `max_iterations`; reaching the limit returns the
latest labels rather than claiming convergence. This bound is required because
the original simultaneous rule does not guarantee convergence. Empty input is
a converged `empty` result with zero iterations. Singletons, all-identical
points, disconnected neighborhoods, arbitrary integer labels, and repeated
states are otherwise processed by the same rules.

## Complexity and boundaries

For `N` observations, `D` features, `E` undirected neighbor pairs, `K` labels,
and `I` completed iterations, neighborhood construction takes `O(N^2 D)` time.
Voting takes `O(I (N + E + N K))` time in this intentionally clear oracle.
The adjacency lists and working state use `O(N + E + D + K)` memory; the
implementation does not construct an `N x N` distance matrix. Python object
overhead is not included in that asymptotic bound.

The contract supports arbitrary `int64` label values, applies the mathematical
source's incumbent tie rule, retains isolated labels, and reports bounded
non-convergence explicitly. The reference is standalone: it does not perform
waveform handling, PCA, automatic radius selection, seed selection, or
post-sorting.

## Native CPU implementation

The same contract is implemented in `neurale::sorting::valley_seeking` and the
raw kernel is exposed only through the private native sorting namespace. The
typed public `neurale.sorting.valley_seeking` wrapper accepts a `FeatureMatrix`
and returns immutable `ValleySeekingResult` diagnostics. The binding requires
exact C-contiguous `float64` features and C-contiguous `int64` labels; it
performs no conversion and releases the GIL during the kernel call.

The native kernel builds a compressed sparse row adjacency list in two pairwise
passes. It has no process-global cache or mutable state and never constructs an
`N x N` distance matrix. Default hard limits are 100,000 observations, 1,024
features, 100,000 distinct labels, and 4,000,000 undirected neighbor pairs.
C++ callers may supply stricter positive `ValleySeekingLimits`; the private
binding uses the defaults and publishes their values as private attributes.
The returned `workspace_bytes` describes the bounded CSR/voting workspace, and
`neighbor_pairs` reports the realized adjacency size. The public wrapper does
not expose the raw binding, estimate a radius, or choose a provider. When the
caller omits initial labels, it uses the documented non-heuristic seed
`initial_labels[i] = i`.
