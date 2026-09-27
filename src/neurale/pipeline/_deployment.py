# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Export fitted decoders to paradigm-independent native pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neurale.exceptions import ValidationError

if TYPE_CHECKING:
    from neurale.decoding import BaseDecoder
    from neurale.streaming import StreamSchema

    from ._stages import KalmanDecoderStage, LdaDecoderStage, LinearDecoderStage


def require_native_decoder(decoder: Any) -> None:
    from neurale.decoding import KalmanDecoder, LDADecoder, LinearDecoder

    if not isinstance(decoder, LinearDecoder | KalmanDecoder | LDADecoder):
        raise TypeError("decoder has no native deployment")
    if getattr(decoder, "context", None) is not None:
        raise ValidationError("native decoder deployment does not support temporal context.")


def decoder_stage(
    decoder: BaseDecoder,
    feature_schema: StreamSchema,
    *,
    output_schema_id: int | None = None,
    output_signal_id: int | None = None,
    output_channel_set_id: int | None = None,
    output_physical_unit: str = "dimensionless",
) -> LinearDecoderStage | KalmanDecoderStage | LdaDecoderStage:
    """Snapshot a fitted Linear, Kalman or LDA decoder as a native stage.

    Scaling and feature selection are exported with the model. LDA requires
    integer labels exactly representable in float64; arbitrary string labels
    and temporal context are not supported. No Python inference fallback exists.
    """
    import numpy as np

    from neurale.decoding import KalmanDecoder, LDADecoder, LinearDecoder
    from neurale.pipeline import KalmanDecoderStage, LdaDecoderStage, LinearDecoderStage

    require_native_decoder(decoder)
    if not decoder.is_fitted:
        raise ValidationError("native deployment requires a fitted decoder.")
    signal, descriptor, contract = _feature_contract(feature_schema)
    if output_schema_id is None:
        output_schema_id = feature_schema.id + 1
    if output_signal_id is None:
        output_signal_id = signal.id + 1
    if output_channel_set_id is None:
        output_channel_set_id = signal.channel_set_id + 1
    feature_names = tuple(descriptor.feature_names)
    common = {
        "output_schema_id": output_schema_id,
        "output_signal_id": output_signal_id,
        "output_channel_set_id": output_channel_set_id,
        "output_physical_unit": output_physical_unit,
        "feature_set_id": signal.feature_set_id,
        "fitted_feature_contract": contract,
    }
    if isinstance(decoder, LinearDecoder | LDADecoder):
        if decoder.context_ is not None:
            raise ValidationError("native online linear deployment does not support context.")
        selected = tuple(decoder.feature_names_in_)
        if selected != feature_names:
            raise ValidationError("native deployment requires the fitted feature names and order.")
        scaling, center, scale = _scaler_parameters(decoder.scaler_)
        model = decoder._model if isinstance(decoder, LDADecoder) else decoder
        stage_type = LdaDecoderStage if isinstance(decoder, LDADecoder) else LinearDecoderStage
        extra = (
            {"classes": tuple(model.classes_.tolist())} if isinstance(decoder, LDADecoder) else {}
        )
        return stage_type(
            **common,
            selection=tuple(range(len(feature_names))),
            selected_feature_names=selected,
            n_features=len(selected),
            n_outputs=len(model.intercept_),
            coefficients=tuple(np.asarray(model.coef_, dtype=np.float64).ravel()),
            intercept=tuple(np.asarray(model.intercept_, dtype=np.float64).ravel()),
            scaling=scaling,
            scaler_center=center,
            scaler_scale=scale,
            **extra,
        )
    if isinstance(decoder, KalmanDecoder):
        selected = tuple(decoder.selected_feature_names_)
        try:
            selection = tuple(feature_names.index(name) for name in selected)
        except ValueError as exc:
            raise ValidationError("the fitted Kalman feature selection is not deployable.") from exc
        parameters = decoder.parameters_
        scaling, center, scale = _scaler_parameters(decoder.scaler_)
        values = {
            name: tuple(np.asarray(getattr(parameters, f"{name}_"), dtype=np.float64).ravel())
            for name in (
                "transition",
                "transition_offset",
                "observation",
                "observation_offset",
                "process_covariance",
                "observation_covariance",
                "initial_state",
                "initial_covariance",
            )
        }
        return KalmanDecoderStage(
            **common,
            selection=selection,
            selected_feature_names=selected,
            state_dim=parameters.n_states_,
            observation_dim=parameters.n_observations_,
            **values,
            innovation_jitter=decoder.innovation_jitter,
            missing=decoder.missing,
            scaling=scaling,
            scaler_center=center,
            scaler_scale=scale,
        )
    raise TypeError("decoder has no native deployment")


def _feature_contract(feature_schema: Any) -> tuple[Any, Any, Any]:
    from neurale.pipeline import FittedFeatureContract
    from neurale.streaming import FeatureTimestampReference, SignalKind

    signals = tuple(feature_schema.signals)
    if len(signals) != 1 or signals[0].kind != SignalKind.FEATURE:
        raise ValidationError("online deployment requires one native feature signal.")
    signal = signals[0]
    descriptors = tuple(
        value for value in feature_schema.feature_sets if value.id == signal.feature_set_id
    )
    if len(descriptors) != 1:
        raise ValidationError("the feature signal must reference one declared feature set.")
    descriptor = descriptors[0]
    if descriptor.timestamp_reference != FeatureTimestampReference.WINDOW_CENTER:
        raise ValidationError("online deployment requires window-center feature timestamps.")
    unit_symbols = {value.id: value.symbol for value in feature_schema.units}
    try:
        symbols = tuple(unit_symbols[unit_id] for unit_id in descriptor.unit_ids)
    except KeyError as exc:
        raise ValidationError("the feature set references an undeclared unit.") from exc
    contract = FittedFeatureContract(
        feature_names=tuple(descriptor.feature_names),
        feature_unit_symbols=symbols,
        observation_rate_numerator=signal.fs.numerator,
        observation_rate_denominator=signal.fs.denominator,
        window_length_ns=descriptor.window_length_ns,
        shift_ns=descriptor.shift_ns,
        algorithm_name=descriptor.algorithm_name,
        algorithm_version=descriptor.algorithm_version,
        source_stream=descriptor.source_stream,
    )
    return signal, descriptor, contract


def _scaler_parameters(scaler: Any) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    import numpy as np

    from neurale.models import MinMaxScaler, StandardScaler

    if scaler is None:
        return "none", (), ()
    if isinstance(scaler, StandardScaler):
        return (
            "standard",
            tuple(np.asarray(scaler.mean_, dtype=np.float64).ravel()),
            tuple(np.asarray(scaler.scale_, dtype=np.float64).ravel()),
        )
    if isinstance(scaler, MinMaxScaler):
        return (
            "minmax",
            tuple(np.asarray(scaler.min_, dtype=np.float64).ravel()),
            tuple(np.asarray(scaler.scale_, dtype=np.float64).ravel()),
        )
    raise ValidationError("the fitted decoder uses an unsupported scaler.")
