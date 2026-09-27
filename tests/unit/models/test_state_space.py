#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import FilterResult, KalmanFilter, LinearGaussianStateSpace
from neurale.runtime import runtime_context

# --------------------------------------------------------------------------------------
# Independent NumPy reference
# --------------------------------------------------------------------------------------


class _ReferenceKalman:
    """Textbook Kalman recursion used only to check the native implementation.

    The covariance update is the plain ``P - K H P`` form rather than the Joseph
    form the production filter uses, so agreement is evidence about the result
    and not about a shared derivation.
    """

    def __init__(self, model: LinearGaussianStateSpace, state, covariance, jitter: float = 0.0):
        self.transition = np.array(model.transition_)
        self.transition_offset = np.array(model.transition_offset_)
        self.observation = np.array(model.observation_)
        self.observation_offset = np.array(model.observation_offset_)
        self.process_covariance = np.array(model.process_covariance_)
        self.observation_covariance = np.array(model.observation_covariance_)
        self.state = np.array(state, dtype=np.float64)
        self.covariance = np.array(covariance, dtype=np.float64)
        self.jitter = jitter

    def predict(self) -> None:
        self.state = self.transition @ self.state + self.transition_offset
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + self.process_covariance
        )

    def update(self, observation: np.ndarray) -> None:
        innovation = observation - (self.observation @ self.state + self.observation_offset)
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.observation_covariance
            + self.jitter * np.eye(self.observation.shape[0])
        )
        gain = np.linalg.solve(innovation_covariance, self.observation @ self.covariance).T
        self.state = self.state + gain @ innovation
        self.covariance = self.covariance - gain @ self.observation @ self.covariance

    def filter(self, observations: np.ndarray, observed=None):
        states = []
        covariances = []
        for i, row in enumerate(observations):
            self.predict()
            if observed is None or observed[i]:
                self.update(row)
            states.append(self.state.copy())
            covariances.append(self.covariance.copy())
        return np.array(states), np.array(covariances)


# --------------------------------------------------------------------------------------
# Synthetic systems
# --------------------------------------------------------------------------------------

_TRANSITION = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.7]])
_TRANSITION_OFFSET = np.array([0.05, -0.02, 0.01])
_PROCESS_COVARIANCE = np.diag([0.10, 0.20, 0.15])
_OBSERVATION_COVARIANCE = np.diag([0.10, 0.20, 0.15, 0.12])


def _observation_map(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 3)), rng.normal(size=4)


def _simulate(
    n_samples: int = 600,
    *,
    seed: int = 20260802,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one segment from the reference system defined above."""

    rng = np.random.default_rng(seed)
    observation, observation_offset = _observation_map()
    state = np.zeros(3) if start is None else np.array(start, dtype=np.float64)
    states = np.zeros((n_samples, 3))
    observations = np.zeros((n_samples, 4))
    for i in range(n_samples):
        state = (
            _TRANSITION @ state
            + _TRANSITION_OFFSET
            + rng.multivariate_normal(np.zeros(3), _PROCESS_COVARIANCE)
        )
        states[i] = state
        observations[i] = (
            observation @ state
            + observation_offset
            + rng.multivariate_normal(np.zeros(4), _OBSERVATION_COVARIANCE)
        )
    return states, observations


def _duplicated_channel_system() -> tuple[np.ndarray, np.ndarray]:
    """A system whose second observation channel repeats the first.

    Two bitwise-identical target columns give bitwise-identical fitted rows and
    residuals, so the estimated observation covariance has two identical rows and
    is exactly singular. The innovation covariance inherits that singularity.
    """

    states, observations = _simulate(300, seed=31)
    observations[:, 1] = observations[:, 0]
    return states, observations


def _scalar_system(gain: float, n_samples: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A noiseless 1D system whose state scales by ``gain`` each step.

    The state is observed directly, so the fitted transition is ``[[gain]]`` and
    the fitted observation is ``[[1.0]]``.
    """

    states = (gain ** np.arange(n_samples, dtype=np.float64)).reshape(-1, 1)
    return states, states.copy()


def _weakly_observed_system(
    n_samples: int = 400,
    *,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """A mean-reverting state seen through heavy observation noise.

    One measurement says little about the state, so the filter leans on its
    prior and the choice of prior for the first row stays visible in the output.
    """

    rng = np.random.default_rng(seed)
    state = 0.0
    states = np.zeros((n_samples, 1))
    observations = np.zeros((n_samples, 1))
    for i in range(n_samples):
        state = 0.5 * state + rng.normal(0.0, 1.0)
        states[i] = state
        observations[i] = state + rng.normal(0.0, 5.0)
    return states, observations


def _fitted(**kwargs) -> tuple[LinearGaussianStateSpace, np.ndarray, np.ndarray]:
    states, observations = _simulate()
    model = LinearGaussianStateSpace(**kwargs).fit(states, observations)
    return model, states, observations


# --------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------


def test_estimation_recovers_generating_system() -> None:
    model, states, _ = _fitted()
    observation, observation_offset = _observation_map()

    assert model.n_samples_ == states.shape[0]
    assert model.n_segments_ == 1
    assert model.n_transitions_ == states.shape[0] - 1
    assert model.n_states_ == 3
    assert model.n_observations_ == 4
    assert np.allclose(model.transition_, _TRANSITION, atol=0.05)
    assert np.allclose(model.transition_offset_, _TRANSITION_OFFSET, atol=0.05)
    assert np.allclose(model.observation_, observation, atol=0.1)
    assert np.allclose(model.observation_offset_, observation_offset, atol=0.1)
    assert np.allclose(model.process_covariance_, _PROCESS_COVARIANCE, atol=0.05)
    assert np.allclose(model.observation_covariance_, _OBSERVATION_COVARIANCE, atol=0.05)


def test_estimated_maps_match_direct_least_squares() -> None:
    model, states, observations = _fitted()
    transition_design = np.column_stack([states[:-1], np.ones(states.shape[0] - 1)])
    transition_solution, *_ = np.linalg.lstsq(transition_design, states[1:], rcond=None)
    observation_design = np.column_stack([states, np.ones(states.shape[0])])
    observation_solution, *_ = np.linalg.lstsq(observation_design, observations, rcond=None)

    assert np.allclose(model.transition_, transition_solution[:-1].T)
    assert np.allclose(model.transition_offset_, transition_solution[-1])
    assert np.allclose(model.observation_, observation_solution[:-1].T)
    assert np.allclose(model.observation_offset_, observation_solution[-1])


def test_estimated_covariances_are_residual_ml_covariances() -> None:
    model, states, observations = _fitted()
    process_residuals = states[1:] - (states[:-1] @ model.transition_.T + model.transition_offset_)
    observation_residuals = observations - (
        states @ model.observation_.T + model.observation_offset_
    )

    assert np.allclose(
        model.process_covariance_,
        process_residuals.T @ process_residuals / process_residuals.shape[0],
    )
    assert np.allclose(
        model.observation_covariance_,
        observation_residuals.T @ observation_residuals / observation_residuals.shape[0],
    )


def test_estimated_covariances_are_symmetric_psd() -> None:
    model, _, _ = _fitted()

    for covariance in (model.process_covariance_, model.observation_covariance_):
        assert np.array_equal(covariance, covariance.T)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_fit_without_offsets_leaves_both_offsets_at_zero() -> None:
    states, observations = _simulate()

    model = LinearGaussianStateSpace(fit_offsets=False).fit(states, observations)

    assert np.array_equal(model.transition_offset_, np.zeros(3))
    assert np.array_equal(model.observation_offset_, np.zeros(4))


def test_jitter_is_added_only_where_requested() -> None:
    states, observations = _simulate()
    plain = LinearGaussianStateSpace().fit(states, observations)
    jittered = LinearGaussianStateSpace(jitter=0.5).fit(states, observations)

    assert np.allclose(
        jittered.process_covariance_ - plain.process_covariance_,
        0.5 * np.eye(3),
    )
    assert np.allclose(
        jittered.observation_covariance_ - plain.observation_covariance_,
        0.5 * np.eye(4),
    )
    assert np.allclose(
        jittered.initial_covariance_ - plain.initial_covariance_,
        0.5 * np.eye(3),
    )
    assert np.allclose(jittered.transition_, plain.transition_)


def test_single_segment_yields_zero_initial_covariance() -> None:
    model, states, _ = _fitted()

    assert np.array_equal(model.initial_state_, states[0])
    assert np.array_equal(model.initial_covariance_, np.zeros((3, 3)))


def test_fitted_parameters_are_read_only() -> None:
    model, _, _ = _fitted()

    for parameter in (
        model.transition_,
        model.transition_offset_,
        model.observation_,
        model.observation_offset_,
        model.process_covariance_,
        model.observation_covariance_,
        model.initial_state_,
        model.initial_covariance_,
    ):
        assert not parameter.flags.writeable
        with pytest.raises(ValueError):
            parameter.reshape(-1)[0] = 0.0


# --------------------------------------------------------------------------------------
# Parameter snapshots and refitting
# --------------------------------------------------------------------------------------


def test_fit_publishes_new_snapshot() -> None:
    model = LinearGaussianStateSpace(fit_offsets=False)
    model.fit(*_scalar_system(2.0))
    first = model.parameters_

    model.fit(*_scalar_system(3.0))

    assert model.parameters_ is not first
    assert np.allclose(first.transition_, [[2.0]])
    assert np.allclose(model.parameters_.transition_, [[3.0]])


def test_parameter_snapshot_is_immutable() -> None:
    model, _, _ = _fitted()
    parameters = model.parameters_

    with pytest.raises(AttributeError, match="immutable"):
        parameters.transition_ = np.eye(3)
    with pytest.raises(AttributeError, match="immutable"):
        parameters._native = None
    with pytest.raises(AttributeError, match="immutable"):
        del parameters._native
    for arr in (parameters.transition_, parameters.process_covariance_):
        assert not arr.flags.writeable


def test_refit_leaves_existing_filter_parameters() -> None:
    model = LinearGaussianStateSpace(fit_offsets=False)
    model.fit(*_scalar_system(2.0))
    filter_ = KalmanFilter(model, initial_state=np.array([1.0]))
    snapshot = filter_.parameters

    model.fit(*_scalar_system(3.0))

    # The estimator moved on; the filter did not, and it says so.
    assert np.allclose(model.transition_, [[3.0]])
    assert filter_.parameters is snapshot
    assert filter_.parameters is not model.parameters_
    assert np.allclose(filter_.parameters.transition_, [[2.0]])
    assert np.allclose(filter_.predict(), [2.0])
    # A filter built afterwards runs the new parameters.
    assert np.allclose(KalmanFilter(model, initial_state=np.array([1.0])).predict(), [3.0])


def test_refit_with_new_state_dimension_keeps_filter_consistent() -> None:
    model = LinearGaussianStateSpace(fit_offsets=False)
    model.fit(*_scalar_system(2.0))
    filter_ = KalmanFilter(model, initial_state=np.array([1.0]))

    model.fit(*_simulate(100))

    assert model.n_states_ == 3
    assert filter_.n_states == 1
    assert filter_.n_observations == 1
    assert filter_.state.shape == (1,)
    assert filter_.covariance.shape == (1, 1)
    assert np.allclose(filter_.predict(), [2.0])
    # The dimensions the filter validates against are its own, not the estimator's.
    with pytest.raises(ValidationError, match="observation must have shape"):
        filter_.update(np.zeros(4))
    with pytest.raises(ValidationError, match="state must have shape"):
        filter_.reset(np.zeros(3))


def test_filters_from_separate_fits_are_independent() -> None:
    model = LinearGaussianStateSpace(fit_offsets=False)
    model.fit(*_scalar_system(2.0))
    doubling = KalmanFilter(model, initial_state=np.array([1.0]))
    model.fit(*_scalar_system(3.0))
    tripling = KalmanFilter(model, initial_state=np.array([1.0]))

    for _ in range(3):
        doubling.predict()
        tripling.predict()

    assert np.allclose(doubling.state, [8.0])
    assert np.allclose(tripling.state, [27.0])


def test_filter_builds_from_snapshot_alone() -> None:
    model, _, observations = _fitted()
    parameters = model.parameters_

    filter_ = KalmanFilter(parameters, initial_covariance=np.eye(3))

    assert filter_.parameters is parameters
    assert filter_.device_ == "cpu"
    assert filter_.n_states == 3
    assert filter_.n_observations == 4
    expected = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:20])
    assert np.array_equal(filter_.filter(observations[:20]).states, expected.states)


# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------


def _two_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_states, first_observations = _simulate(200, seed=1)
    second_states, second_observations = _simulate(
        150,
        seed=2,
        start=np.array([50.0, -40.0, 30.0]),
    )
    states = np.vstack([first_states, second_states])
    observations = np.vstack([first_observations, second_observations])
    return states, observations, np.array([200, 150])


def test_segments_exclude_boundary_transition() -> None:
    states, observations, lengths = _two_segments()

    model = LinearGaussianStateSpace().fit(states, observations, segment_lengths=lengths)

    assert model.n_segments_ == 2
    assert model.n_transitions_ == states.shape[0] - 2
    # The boundary jump is a huge outlier, so treating the rows as one segment
    # visibly distorts the transition fit.
    single = LinearGaussianStateSpace().fit(states, observations)
    assert single.n_transitions_ == states.shape[0] - 1
    assert np.allclose(model.transition_, _TRANSITION, atol=0.1)
    assert not np.allclose(single.transition_, _TRANSITION, atol=0.1)


def test_segments_define_initial_state_and_covariance() -> None:
    states, observations, lengths = _two_segments()
    starts = np.vstack([states[0], states[200]])

    model = LinearGaussianStateSpace().fit(states, observations, segment_lengths=lengths)

    assert np.allclose(model.initial_state_, starts.mean(axis=0))
    deviations = starts - starts.mean(axis=0)
    assert np.allclose(model.initial_covariance_, deviations.T @ deviations / 2.0)


def test_segment_transition_pairs_match_direct_stack() -> None:
    states, observations, lengths = _two_segments()
    previous = np.vstack([states[0:199], states[200:349]])
    following = np.vstack([states[1:200], states[201:350]])
    design = np.column_stack([previous, np.ones(previous.shape[0])])
    solution, *_ = np.linalg.lstsq(design, following, rcond=None)

    model = LinearGaussianStateSpace().fit(states, observations, segment_lengths=lengths)

    assert np.allclose(model.transition_, solution[:-1].T)
    assert np.allclose(model.transition_offset_, solution[-1])


@pytest.mark.parametrize(
    ("lengths", "message"),
    [
        (np.array([100, 100]), "must sum to the number of samples"),
        (np.array([600, 1]), "must sum to the number of samples"),
        (np.array([0, 600]), "must be positive"),
        (np.array([-1, 601]), "must be positive"),
        (np.array([600.0]), "must contain integers"),
        (np.array([[300], [300]]), "must be a non-empty 1D array"),
        (np.array([], dtype=int), "must be a non-empty 1D array"),
        ([300, 300], "must be a numpy.ndarray"),
    ],
    ids=["short", "long", "zero", "negative", "float", "2d", "empty", "list"],
)
def test_fit_rejects_invalid_segment_lengths(lengths, message: str) -> None:
    states, observations = _simulate()

    with pytest.raises(ValidationError, match=message):
        LinearGaussianStateSpace().fit(states, observations, segment_lengths=lengths)


def test_fit_requires_at_least_one_transition() -> None:
    states, observations = _simulate(3)

    with pytest.raises(ValidationError, match="at least one segment must contain two samples"):
        LinearGaussianStateSpace().fit(
            states,
            observations,
            segment_lengths=np.array([1, 1, 1]),
        )


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------


def test_single_step_predict_and_update_match_reference() -> None:
    model, _, observations = _fitted()
    covariance = np.eye(3)
    filter_ = KalmanFilter(model, initial_covariance=covariance)
    reference = _ReferenceKalman(model, model.initial_state_, covariance)

    for row in observations[:25]:
        filter_.predict()
        reference.predict()
        assert np.allclose(filter_.state, reference.state)
        assert np.allclose(filter_.covariance, reference.covariance)

        filter_.update(row)
        reference.update(row)
        assert np.allclose(filter_.state, reference.state)
        assert np.allclose(filter_.covariance, reference.covariance)


def test_batch_filtering_matches_reference() -> None:
    model, states, observations = _fitted()
    covariance = np.eye(3)
    filter_ = KalmanFilter(model, initial_covariance=covariance)
    reference = _ReferenceKalman(model, model.initial_state_, covariance)

    result = filter_.filter(observations)
    reference_states, reference_covariances = reference.filter(observations)

    assert isinstance(result, FilterResult)
    assert result.states.shape == (observations.shape[0], 3)
    assert result.covariances.shape == (observations.shape[0], 3, 3)
    assert np.allclose(result.states, reference_states)
    assert np.allclose(result.covariances, reference_covariances)
    # The filter tracks the latent state better than a memoryless decode of the
    # same observations, which is the point of carrying the transition model.
    memoryless = (observations - model.observation_offset_) @ np.linalg.pinv(model.observation_).T
    assert np.sqrt(((result.states - states) ** 2).mean()) < np.sqrt(
        ((memoryless - states) ** 2).mean()
    )


def test_batch_filtering_equals_repeated_single_steps() -> None:
    """The frozen default contract: the state is the posterior at t-1, each row is z[t]."""

    model, _, observations = _fitted()
    batch = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:40])

    stepwise = KalmanFilter(model, initial_covariance=np.eye(3))
    states = []
    covariances = []
    for row in observations[:40]:
        stepwise.predict()
        stepwise.update(row)
        states.append(stepwise.state)
        covariances.append(stepwise.covariance)

    assert np.array_equal(batch.states, np.array(states))
    assert np.array_equal(batch.covariances, np.array(covariances))


def test_predict_first_false_corrects_first_row_in_place() -> None:
    """The other contract: the state is already the prior for the first row."""

    model, _, observations = _fitted()
    batch = KalmanFilter(model, initial_covariance=np.eye(3)).filter(
        observations[:12],
        predict_first=False,
    )

    stepwise = KalmanFilter(model, initial_covariance=np.eye(3))
    stepwise.update(observations[0])
    states = [stepwise.state]
    covariances = [stepwise.covariance]
    for row in observations[1:12]:
        stepwise.predict()
        stepwise.update(row)
        states.append(stepwise.state)
        covariances.append(stepwise.covariance)

    assert np.array_equal(batch.states, np.array(states))
    assert np.array_equal(batch.covariances, np.array(covariances))
    assert np.array_equal(batch.states[-1], stepwise.state)


def test_predict_first_uses_prior_on_first_row_only() -> None:
    """``initial_state_`` estimates the first state of a segment, not the one before it."""

    states, observations = _weakly_observed_system()
    model = LinearGaussianStateSpace(fit_offsets=False).fit(states, observations)
    # A confident initial covariance makes the choice of prior visible; the
    # observations are too noisy to overrule it on the first row.
    covariance = np.eye(1) * 1e-6

    aligned = KalmanFilter(model, initial_covariance=covariance).filter(
        observations,
        predict_first=False,
    )
    advanced = KalmanFilter(model, initial_covariance=covariance).filter(observations)

    assert np.array_equal(model.initial_state_, states[0])
    # The first row measures the state the filter is already holding, so only
    # the pass that does not advance it first reproduces that state.
    assert abs(aligned.states[0, 0] - states[0, 0]) < 1e-6
    assert abs(advanced.states[0, 0] - states[0, 0]) > 1e-2
    # The flag picks the prior of the first row and nothing else: every row is
    # still the posterior at its own measurement, and the disagreement decays.
    assert np.allclose(aligned.states[50:], advanced.states[50:], rtol=0.0, atol=1e-9)


def test_predict_first_false_consumes_exactly_one_step() -> None:
    """After the first row, the default contract continues the same timeline."""

    model, _, observations = _fitted()
    covariance = np.eye(3)
    whole = KalmanFilter(model, initial_covariance=covariance).filter(
        observations[:20],
        predict_first=False,
    )

    split = KalmanFilter(model, initial_covariance=covariance)
    head = split.filter(observations[:1], predict_first=False)
    tail = split.filter(observations[1:20])

    assert np.array_equal(np.vstack([head.states, tail.states]), whole.states)


def test_predict_first_false_on_single_row_only_updates() -> None:
    model, _, observations = _fitted()
    batch = KalmanFilter(model, initial_covariance=np.eye(3)).filter(
        observations[:1],
        predict_first=False,
    )

    stepwise = KalmanFilter(model, initial_covariance=np.eye(3))
    stepwise.update(observations[0])

    assert np.array_equal(batch.states[0], stepwise.state)
    assert np.array_equal(batch.covariances[0], stepwise.covariance)


def test_predict_first_false_leaves_missing_first_row() -> None:
    model, _, observations = _fitted()
    damaged = observations[:6].copy()
    damaged[0] = np.nan
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    result = filter_.filter(damaged, missing="predict", predict_first=False)

    # No measurement and no transition: the first row reports the state as it was.
    assert np.array_equal(result.states[0], model.initial_state_)
    assert np.array_equal(result.covariances[0], np.eye(3))


def test_predict_first_must_be_bool() -> None:
    model, _, observations = _fitted()

    with pytest.raises(ValidationError, match="predict_first must be a bool"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations, predict_first=1)


def test_filter_leaves_final_posterior_in_state() -> None:
    model, _, observations = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    result = filter_.filter(observations[:30])

    assert np.array_equal(filter_.state, result.states[-1])
    assert np.array_equal(filter_.covariance, result.covariances[-1])


def test_filtered_covariances_stay_symmetric() -> None:
    model, _, observations = _fitted()

    result = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:50])

    for covariance in result.covariances:
        assert np.array_equal(covariance, covariance.T)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_filter_result_arrays_are_read_only() -> None:
    model, _, observations = _fitted()

    result = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:10])

    assert not result.states.flags.writeable
    assert not result.covariances.flags.writeable


def test_filtering_does_not_mutate_fitted_parameters() -> None:
    model, _, observations = _fitted()
    before = {
        name: np.array(getattr(model, name))
        for name in (
            "transition_",
            "transition_offset_",
            "observation_",
            "observation_offset_",
            "process_covariance_",
            "observation_covariance_",
            "initial_state_",
            "initial_covariance_",
        )
    }

    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))
    filter_.filter(observations)
    filter_.predict()
    filter_.update(observations[0])

    for name, value in before.items():
        assert np.array_equal(getattr(model, name), value)


def test_filters_share_parameters_not_state() -> None:
    model, _, observations = _fitted()
    first = KalmanFilter(model, initial_covariance=np.eye(3))
    second = KalmanFilter(model, initial_covariance=np.eye(3))

    first.filter(observations[:20])

    assert np.array_equal(second.state, model.initial_state_)
    assert np.array_equal(second.covariance, np.eye(3))
    assert not np.array_equal(first.state, second.state)


# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------


def test_reset_restores_configured_initial_state() -> None:
    model, _, observations = _fitted()
    covariance = np.eye(3) * 2.0
    filter_ = KalmanFilter(model, initial_covariance=covariance)

    first = filter_.filter(observations[:30])
    filter_.reset()

    assert np.array_equal(filter_.state, model.initial_state_)
    assert np.array_equal(filter_.covariance, covariance)
    assert np.array_equal(filter_.filter(observations[:30]).states, first.states)


def test_reset_accepts_caller_supplied_values() -> None:
    model, _, _ = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))
    state = np.array([1.0, 2.0, 3.0])
    covariance = np.diag([4.0, 5.0, 6.0])

    filter_.predict()
    filter_.reset(state, covariance)

    assert np.array_equal(filter_.state, state)
    assert np.array_equal(filter_.covariance, covariance)
    # The configured defaults are unchanged, so a plain reset still restores them.
    filter_.reset()
    assert np.array_equal(filter_.state, model.initial_state_)
    assert np.array_equal(filter_.covariance, np.eye(3))


def test_filter_defaults_to_fitted_initial_values() -> None:
    model, _, _ = _fitted()

    filter_ = KalmanFilter(model)

    assert np.array_equal(filter_.state, model.initial_state_)
    assert np.array_equal(filter_.covariance, model.initial_covariance_)
    assert np.array_equal(filter_.initial_state, model.initial_state_)
    assert np.array_equal(filter_.initial_covariance, model.initial_covariance_)


def test_initial_values_are_copied_from_source() -> None:
    model, _, _ = _fitted()
    state = np.array([1.0, 2.0, 3.0])
    covariance = np.eye(3)

    filter_ = KalmanFilter(model, initial_state=state, initial_covariance=covariance)
    state[0] = 99.0
    covariance[0, 0] = 99.0

    assert np.array_equal(filter_.state, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(filter_.covariance, np.eye(3))
    assert not filter_.initial_state.flags.writeable
    assert not filter_.initial_covariance.flags.writeable


# --------------------------------------------------------------------------------------
# Missing observations
# --------------------------------------------------------------------------------------


def test_filter_rejects_non_finite_rows_by_default() -> None:
    model, _, observations = _fitted()
    damaged = observations.copy()
    damaged[5] = np.nan

    with pytest.raises(ValidationError, match="must contain finite values"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(damaged)


def test_missing_predict_runs_predict_step_alone() -> None:
    model, _, observations = _fitted()
    damaged = observations.copy()
    damaged[5] = np.nan
    damaged[9] = np.nan
    observed = np.ones(observations.shape[0], dtype=bool)
    observed[[5, 9]] = False
    covariance = np.eye(3)

    result = KalmanFilter(model, initial_covariance=covariance).filter(
        damaged,
        missing="predict",
    )
    reference = _ReferenceKalman(model, model.initial_state_, covariance)
    reference_states, reference_covariances = reference.filter(damaged, observed)

    assert np.allclose(result.states, reference_states)
    assert np.allclose(result.covariances, reference_covariances)
    # A skipped update leaves the state less certain than the corrected one.
    assert np.trace(result.covariances[5]) > np.trace(result.covariances[4])


def test_missing_predict_accepts_fully_missing_batch() -> None:
    model, _, observations = _fitted()
    missing = np.full_like(observations[:10], np.nan)
    covariance = np.eye(3)

    result = KalmanFilter(model, initial_covariance=covariance).filter(missing, missing="predict")
    reference = _ReferenceKalman(model, model.initial_state_, covariance)
    reference_states, reference_covariances = reference.filter(
        missing,
        np.zeros(10, dtype=bool),
    )

    assert np.allclose(result.states, reference_states)
    assert np.allclose(result.covariances, reference_covariances)


def test_missing_predict_rejects_partially_missing_rows() -> None:
    model, _, observations = _fitted()
    damaged = observations.copy()
    damaged[5, 0] = np.nan

    with pytest.raises(ValidationError, match="partially missing observation rows"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(damaged, missing="predict")


@pytest.mark.parametrize(
    "row",
    [np.inf, -np.inf, np.array([np.inf, 1.0, 2.0, 3.0]), np.array([np.nan, np.inf, 1.0, 2.0])],
    ids=["positive", "negative", "single-entry", "with-nan"],
)
def test_missing_predict_rejects_infinities(row) -> None:
    """An infinity is an upstream fault, not a measurement that was never taken."""

    model, _, observations = _fitted()
    damaged = observations.copy()
    damaged[5] = row

    with pytest.raises(ValidationError, match="must not contain infinite values"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(damaged, missing="predict")


def test_update_rejects_missing_observation() -> None:
    model, _, _ = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    with pytest.raises(ValidationError, match="must contain finite values"):
        filter_.update(np.full(4, np.nan))


@pytest.mark.parametrize("policy", ["skip", "", None, 0], ids=["unknown", "empty", "none", "zero"])
def test_filter_rejects_unknown_missing_policy(policy) -> None:
    model, _, observations = _fitted()

    with pytest.raises(ValidationError, match="missing must be one of"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations, missing=policy)


# --------------------------------------------------------------------------------------
# Singular and ill-conditioned covariances
# --------------------------------------------------------------------------------------


def test_singular_innovation_covariance_is_reported() -> None:
    states, observations = _duplicated_channel_system()
    model = LinearGaussianStateSpace().fit(states, observations)

    assert np.array_equal(model.observation_covariance_[0], model.observation_covariance_[1])
    assert np.linalg.matrix_rank(model.observation_covariance_) < 4
    with pytest.raises(ValidationError, match="innovation covariance is singular"):
        KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations)


def test_explicit_jitter_recovers_singular_innovation() -> None:
    states, observations = _duplicated_channel_system()
    model = LinearGaussianStateSpace().fit(states, observations)
    covariance = np.eye(3)

    result = KalmanFilter(model, initial_covariance=covariance, jitter=1e-6).filter(observations)
    reference = _ReferenceKalman(model, model.initial_state_, covariance, jitter=1e-6)
    reference_states, reference_covariances = reference.filter(observations)

    assert np.all(np.isfinite(result.states))
    # A jittered gain is no longer the optimal gain, and the two covariance
    # recursions are only algebraically equal for the optimal one, so they agree
    # to the order of the jitter rather than to roundoff. The Joseph form the
    # production filter uses is the one that stays valid for any gain.
    assert np.allclose(result.states, reference_states, atol=1e-4)
    assert np.allclose(result.covariances, reference_covariances, atol=1e-4)
    for covariance in result.covariances:
        assert np.array_equal(covariance, covariance.T)


def test_fit_jitter_makes_filter_solvable() -> None:
    states, observations = _duplicated_channel_system()
    model = LinearGaussianStateSpace(jitter=1e-6).fit(states, observations)

    result = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations)

    assert np.all(np.isfinite(result.states))


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[1.0, 0.5], [0.4, 1.0]]), "must have shape"),
        (np.array([[1.0, 0.5, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]), "must be symmetric"),
        (np.diag([1.0, -1.0, 1.0]), "must be positive semidefinite"),
        (np.full((3, 3), np.nan), "must contain finite values"),
        (np.zeros(3), "must be a 2D array"),
        ([[1.0]], "must be a numpy.ndarray"),
    ],
    ids=["shape", "asymmetric", "indefinite", "nan", "1d", "not-array"],
)
def test_initial_covariance_is_validated(covariance, message: str) -> None:
    model, _, _ = _fitted()

    with pytest.raises(ValidationError, match=message):
        KalmanFilter(model, initial_covariance=covariance)


def test_reset_validates_caller_supplied_values() -> None:
    model, _, _ = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    with pytest.raises(ValidationError, match="state must have shape"):
        filter_.reset(np.zeros(2))
    with pytest.raises(ValidationError, match="must be positive semidefinite"):
        filter_.reset(covariance=np.diag([1.0, -1.0, 1.0]))
    # A rejected reset leaves the filter untouched.
    assert np.array_equal(filter_.state, model.initial_state_)


def test_psd_initial_covariance_is_accepted() -> None:
    model, _, _ = _fitted()
    singular = np.diag([1.0, 0.0, 2.0])

    filter_ = KalmanFilter(model, initial_covariance=singular)

    assert np.array_equal(filter_.covariance, singular)


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1e-12, 1e-12, -9e-9]),
        np.diag([1e-12, 1e-12, -1e-9]),
        np.diag([1.0, 1.0, -1e-12]),
        1e-30 * np.diag([1.0, 1.0, -1.0]),
    ],
    ids=["small-scale", "smaller-negative", "unit-scale", "tiny-scale"],
)
def test_definiteness_is_judged_against_matrix_scale(covariance) -> None:
    """A negative eigenvalue is measured against the matrix, not against 1.0.

    Every matrix here used to pass: the tolerance was floored at ``1.0``, so for
    a small covariance the check degenerated into an absolute ``1e-8`` slack and
    accepted a negative eigenvalue orders of magnitude above the positive ones.
    """

    model, _, _ = _fitted()

    with pytest.raises(ValidationError, match="must be positive semidefinite"):
        KalmanFilter(model, initial_covariance=covariance)


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1e-12, 2e-12, 3e-12]),
        1e-30 * np.eye(3),
        np.diag([1e-9, 1e-9, 0.0]),
        # Roundoff-level negativity, the case the tolerance exists for.
        np.diag([1.0, 1.0, -1e-16]),
    ],
    ids=["small", "tiny", "small-singular", "roundoff"],
)
def test_small_covariance_is_accepted_on_own_scale(covariance) -> None:
    model, _, _ = _fitted()

    filter_ = KalmanFilter(model, initial_covariance=covariance)

    assert np.allclose(filter_.covariance, covariance)


def test_symmetry_is_judged_against_matrix_scale() -> None:
    model, _, _ = _fitted()
    asymmetric = 1e-9 * np.array([[1.0, 0.5, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(ValidationError, match="must be symmetric"):
        KalmanFilter(model, initial_covariance=asymmetric)


# --------------------------------------------------------------------------------------
# Dimensions, fitted state, and the device contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("states", "observations", "message"),
    [
        (np.zeros((0, 3)), np.zeros((0, 4)), "must not be empty"),
        (np.zeros((10, 0)), np.zeros((10, 4)), "must not be empty"),
        (np.zeros((10, 3)), np.zeros((10, 0)), "must not be empty"),
        (np.zeros((10, 3)), np.zeros((9, 4)), "same number of samples"),
        (np.full((10, 3), np.nan), np.zeros((10, 4)), "must contain finite values"),
        (np.zeros((10, 3)), np.full((10, 4), np.inf), "must contain finite values"),
        (np.zeros(10), np.zeros((10, 4)), "must be a 2D array"),
        (np.zeros((10, 3)), np.zeros(10), "must be a 2D array"),
        ([[0.0]], np.zeros((1, 4)), "must be a numpy.ndarray"),
    ],
    ids=[
        "no-rows",
        "no-state-columns",
        "no-observation-columns",
        "mismatched",
        "nan-states",
        "inf-observations",
        "1d-states",
        "1d-observations",
        "not-array",
    ],
)
def test_fit_rejects_invalid_sequences(states, observations, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LinearGaussianStateSpace().fit(states, observations)


@pytest.mark.parametrize(
    ("jitter", "message"),
    [(-1.0, "jitter"), (np.nan, "jitter"), (np.inf, "jitter"), (None, "jitter")],
    ids=["negative", "nan", "inf", "none"],
)
def test_jitter_is_validated(jitter, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LinearGaussianStateSpace(jitter=jitter)


def test_fit_offsets_must_be_bool() -> None:
    with pytest.raises(ValidationError, match="fit_offsets must be a bool"):
        LinearGaussianStateSpace(fit_offsets=1)


def test_estimator_rejects_use_before_fit() -> None:
    model = LinearGaussianStateSpace()

    for attribute in (
        "parameters_",
        "device_",
        "n_samples_",
        "n_segments_",
        "n_transitions_",
        "n_states_",
        "n_observations_",
        "transition_",
        "transition_offset_",
        "observation_",
        "observation_offset_",
        "process_covariance_",
        "observation_covariance_",
        "initial_state_",
        "initial_covariance_",
    ):
        with pytest.raises(ValidationError, match="is not fitted"):
            getattr(model, attribute)
    with pytest.raises(ValidationError, match="is not fitted"):
        KalmanFilter(model)


def test_filter_requires_state_space_model() -> None:
    with pytest.raises(ValidationError, match="model must be a LinearGaussianStateSpace"):
        KalmanFilter(object())


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (np.zeros((10, 3)), "same number of columns"),
        (np.zeros((10, 5)), "same number of columns"),
        (np.zeros((0, 4)), "must not be empty"),
        (np.zeros(4), "must be a 2D array"),
        ([[0.0] * 4], "must be a numpy.ndarray"),
    ],
    ids=["narrow", "wide", "empty", "1d", "not-array"],
)
def test_filter_rejects_invalid_observation_batches(observations, message: str) -> None:
    model, _, _ = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    with pytest.raises(ValidationError, match=message):
        filter_.filter(observations)


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (np.zeros(3), "observation must have shape"),
        (np.zeros((1, 4)), "must be a 2D array|must be a 1D"),
        ([0.0] * 4, "must be a numpy.ndarray"),
    ],
    ids=["wrong-size", "2d", "not-array"],
)
def test_update_rejects_invalid_observations(observation, message: str) -> None:
    model, _, _ = _fitted()
    filter_ = KalmanFilter(model, initial_covariance=np.eye(3))

    with pytest.raises(ValidationError, match=message):
        filter_.update(observation)


def test_estimator_and_filter_report_cpu_device() -> None:
    model, _, _ = _fitted()

    assert model.device_ == "cpu"
    assert KalmanFilter(model).device_ == "cpu"


def test_fit_rejects_explicit_cuda_before_native_dispatch() -> None:
    states, observations = _simulate(20)

    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.state_space.*no CUDA implementation",
        ):
            LinearGaussianStateSpace().fit(states, observations)


@pytest.mark.parametrize("requested", ["auto", "cpu", "cuda"], ids=["auto", "cpu", "cuda"])
def test_ambient_context_does_not_change_fitted_device(requested: str) -> None:
    model, _, observations = _fitted()
    expected = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:20])

    with runtime_context(device=requested):
        assert model.device_ == "cpu"
        replayed = KalmanFilter(model, initial_covariance=np.eye(3)).filter(observations[:20])
        assert np.array_equal(replayed.states, expected.states)
