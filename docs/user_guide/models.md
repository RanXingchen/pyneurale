# Models

## Classification evaluation

`neurale.models.evaluation` owns model-independent `confusion_matrix`,
`accuracy_score`, `precision_score`, `recall_score`, and `f1_score`. These are
also available from `neurale.models`; there is no top-level `neurale.metrics`
package.

For the confusion matrix and unaggregated class scores, an explicit `labels`
sequence fixes the output order. Without it, labels follow stable first
occurrence order: `y_true` first, followed by labels seen only in `y_pred`.
Labels are never sorted. Explicit labels may include absent classes but must
include every observed class. Precision, recall, and F1 use `0.0` when their
denominator is zero and support `None`, `"micro"`, `"macro"`, and `"weighted"`
aggregation.

## CPU-only operations

`PCA`, `LDA`, `LedoitWolfCov`, `dtw`, and the public operations in
`neurale.models.statistics` are CPU-only. Runtime requests for `auto` or `cpu`
use the CPU implementation. An explicit `cuda` request raises
`DeviceUnavailableError`; these operations never fall back silently or enter a
CPU native namespace for a CUDA request.

After a successful fit, `PCA`, `LDA`, and `LedoitWolfCov` expose the read-only
attribute `device_ == "cpu"`. Later inference remains on that fitted device even
inside a different ambient runtime context.

## Locality-preserving projection

`neurale.models.LPP` builds a symmetric binary neighborhood graph with
`neurale.models.knn` and learns a native linear projection. The public model
uses `fit`, `transform`, and `fit_transform`; fitted projection rows are exposed
through `components_`.

`proj_method="LPP"` follows the generalized eigenvalue objective introduced by
Xiaofei He and Partha Niyogi in *Locality Preserving Projections*, NIPS 2003:

<https://papers.nips.cc/paper/2359-locality-preserving-projections>

`proj_method="OPP"` uses the OPP objective: it maximizes the degree-weighted
centered scatter against the regularized graph Laplacian scatter. No
third-party source code is incorporated for either path;
the implementation is original PyNeurale code distributed under the project
MIT license.

An explicit CUDA runtime context runs exact KNN graph construction on the
selected CUDA device. Graph matrix construction, generalized eigensolving, and
`transform` remain on the CPU and share the same implementation as the CPU
path.

## Device-dispatched operations and fitted state

`neurale.models.knn` is stateless and resolves the active runtime device on
each call. It has no fitted device to retain, and an explicit unavailable CUDA
request raises `DeviceUnavailableError` instead of entering the CPU namespace.

`GaussianKDE.fit` creates either a CPU model or a CUDA-resident model and
exposes that actual choice through the read-only `device_` attribute. Later
`pdf`, `logpdf`, and `score_samples` calls execute through the stored native
model; changing the ambient runtime context cannot switch its backend. Copies
and serialized states retain the actual device, normalized fitted bandwidth,
and a read-only copy of the fit samples. Restoration refits an independent
native model on the recorded device. A serialized CUDA model therefore requires
CUDA at restoration and raises `DeviceUnavailableError` when CUDA is
unavailable. If no CUDA ordinal was explicitly selected during the original
fit, restoration uses the then-current/default CUDA-device selection policy.

For `LPP` and its `proj_method="OPP"` mode, CUDA is only a fitting accelerator
for KNN graph discovery. The persistent projection and every `transform` call
are CPU-native, so a fitted estimator exposes `device_ == "cpu"` even when its
neighborhood search ran on CUDA. Copies and serialized states retain the fit
input and reconstruct an independent CPU projection under an internal CPU
context; an ambient CUDA context does not change that reconstruction or later
transforms.
