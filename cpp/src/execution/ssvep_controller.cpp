// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "ssvep_controller.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace neurale::execution
{

using namespace neurale::experiments;
using streaming::StreamStatus;
using namespace ssvep;

namespace
{
std::size_t feature_count(const streaming::StreamSchema& schema)
{
    const auto signals = schema.signals();
    if (signals.size() != 1 || signals[0].kind != streaming::SignalKind::feature ||
        signals[0].dtype != streaming::SignalDType::float64 ||
        signals[0].layout != streaming::SignalLayout::sample_major ||
        signals[0].max_block_samples != 1 || signals[0].n_channels == 0)
        throw std::invalid_argument(
            "SSVEP requires one float64 sample-major feature row per frame");
    return signals[0].n_channels;
}
} // namespace

SSVEPController::SSVEPController(const streaming::StreamSchema& schema, SSVEPConfig task,
                                 std::size_t calibration_trials, IntervalRunner& processing)
    : task_(task), calibration_(calibration_trials), n_features_(feature_count(schema)),
      processing_(processing)
{
    if (ssvep::validate(task_) != ContractStatus::ok || calibration_ < 2U * task_.n_targets ||
        calibration_ % task_.n_targets != 0 || calibration_ >= task_.n_trials)
        throw std::invalid_argument("SSVEP requires balanced calibration and evaluation trials");
    const auto descriptors = schema.feature_sets().descriptors();
    if (descriptors.size() != 1 || descriptors[0].window_length_ns == 0 ||
        descriptors[0].timestamp_reference != streaming::FeatureTimestampReference::window_center)
        throw std::invalid_argument("SSVEP requires window-center feature timestamps");
    window_ns_ = descriptors[0].window_length_ns;
    feature_names_ = descriptors[0].feature_names;
    shift_ns_ = descriptors[0].shift_ns;
    if (shift_ns_ == 0 || window_ns_ > task_.stimulation_duration_ns ||
        shift_ns_ > task_.stimulation_duration_ns - window_ns_)
        throw std::invalid_argument(
            "stimulation must fit a complete feature window and update interval");
    trials_.resize(task_.n_trials);
    traces_.prepare(256);
    displays_.prepare(4096);
    snapshots_.prepare(4096);
}

SSVEPController::~SSVEPController()
{
    cancel();
    if (task_thread_.joinable())
        task_thread_.join();
}

void SSVEPController::publish(SSVEPModelPublication publication)
{
    if (!training() || ready_trials() != calibration_ || published_.load())
        throw std::invalid_argument("SSVEP publishes exactly one model after complete calibration");
    publication_ = std::move(publication);
    published_.store(true, std::memory_order_release);
}

std::span<const double> SSVEPController::features(std::size_t trial) const
{
    if (trial >= ready_trials())
        throw std::out_of_range("SSVEP trial features are not ready");
    return processing_.features(trial);
}

const SSVEPTrial& SSVEPController::trial(std::size_t index) const
{
    if (index >= completed_trials())
        throw std::out_of_range("SSVEP trial is not complete");
    return trials_[index];
}

ContractStatus SSVEPController::validate() const noexcept
{
    return ssvep::validate(task_);
}

StreamStatus SSVEPController::start_execution(ExperimentTimeNs time) noexcept
{
    if (started_ || epoch_ == 0 || processing_.now_ns() == 0)
        return StreamStatus::invalid_state;
    SSVEPStepResult result;
    if (machine_.start(1, task_, time, result) != ContractStatus::ok)
        return StreamStatus::invalid_state;
    started_ = true;
    last_time_ = time;
    const auto status = accept(result);
    if (status != StreamStatus::ok)
        return status;
    try
    {
        const auto processing_status = processing_.start();
        if (processing_status != StreamStatus::ok)
            return processing_status;
        task_thread_ = std::thread([this] { task_main(); });
    }
    catch (...)
    {
        return StreamStatus::invalid_state;
    }
    return StreamStatus::ok;
}

StreamStatus SSVEPController::accept(const SSVEPStepResult& result) noexcept
{
    if (result.trial_decided)
    {
        const auto ordinal = result.trial.record.trial.ordinal;
        if (ordinal < calibration_ && result.trial.record.outcome != TrialOutcome::aborted &&
            (processing_.count(ordinal) == 0 || !result.trial.has_selection))
            return StreamStatus::invalid_frame;
        trials_[ordinal] = result.trial;
        completed_.store(ordinal + 1, std::memory_order_release);
        if (ordinal + 1 == calibration_)
            training_.store(true, std::memory_order_release);
    }
    if ((result.n_transitions || result.n_events || result.trial_decided) &&
        traces_.try_push(result) != StreamStatus::ok)
        return StreamStatus::queue_overflow;
    if (result.snapshot.state == SSVEPState::cue &&
        scheduled_trial_ != result.snapshot.trial.ordinal)
    {
        const auto& snapshot = result.snapshot;
        const auto status = processing_.schedule({snapshot.trial.ordinal + 1,
                                                  epoch_ + snapshot.stimulation.start_ns,
                                                  epoch_ + snapshot.stimulation.end_ns});
        if (status != StreamStatus::ok)
            return status;
        scheduled_trial_ = snapshot.trial.ordinal;
    }
    complete_.store(machine_.complete(), std::memory_order_release);
    return StreamStatus::ok;
}

StreamStatus SSVEPController::advance(ExperimentTimeNs now) noexcept
{
    for (std::size_t guard = 0; guard <= task_.n_trials; ++guard)
    {
        if (training())
        {
            if (!published_.load(std::memory_order_acquire))
            {
                if (machine_.hold_inter_trial_until(now + 1) != ContractStatus::ok)
                    return StreamStatus::invalid_state;
            }
            else
                training_.store(false, std::memory_order_release);
        }
        SSVEPStepResult result;
        if (machine_.step(now, result) != ContractStatus::ok)
            return StreamStatus::invalid_state;
        const auto status = accept(result);
        if (status != StreamStatus::ok || result.settled)
            return status;
    }
    return StreamStatus::invalid_state;
}

void SSVEPController::task_main() noexcept
{
    while (!cancelled_.load(std::memory_order_acquire) && !complete())
    {
        const auto host_now = processing_.now_ns();
        auto value = processing_.status();
        if (host_now < epoch_)
            value = StreamStatus::invalid_frame;
        const auto now = host_now - epoch_;
        if (value == StreamStatus::ok && now < last_time_)
            value = StreamStatus::invalid_frame;
        last_time_ = now;
        if (value == StreamStatus::ok)
            value = advance(now);
        IntervalResult result;
        while (value == StreamStatus::ok && processing_.pop_result(result) == StreamStatus::ok)
            value = apply_result(now, result);
        if (value == StreamStatus::ok)
            value = snapshots_.try_push(machine_.snapshot());
        if (value != StreamStatus::ok)
        {
            task_status_.store(value, std::memory_order_release);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

StreamStatus SSVEPController::apply_result(ExperimentTimeNs now,
                                           const IntervalResult& result) noexcept
{
    const auto snapshot = machine_.snapshot();
    // A delayed result belongs to its original trial, never the next trial.
    if (result.interval.key != snapshot.trial.ordinal + 1 ||
        result.interval.start_ns != epoch_ + snapshot.stimulation.start_ns ||
        result.interval.end_ns != epoch_ + snapshot.stimulation.end_ns ||
        now >= snapshot.decision_deadline_ns ||
        (snapshot.state != SSVEPState::stimulation && snapshot.state != SSVEPState::await_decision))
        return StreamStatus::ok;
    const bool calibration = snapshot.trial.ordinal < calibration_;
    if (result.windows == 0)
        return calibration ? StreamStatus::invalid_frame : StreamStatus::ok;
    double value = snapshot.trial.target_id;
    if (!calibration)
    {
        if (!result.decoded)
            return StreamStatus::invalid_frame;
        const auto frame = result.decoded.view();
        if (frame.blocks.size() != 1 || frame.blocks[0].n_samples != 1 ||
            frame.blocks[0].payload_byte_count != sizeof(value))
            return StreamStatus::invalid_frame;
        std::memcpy(&value, frame.payload.data() + frame.blocks[0].payload_offset, sizeof(value));
    }
    if (!std::isfinite(value) || value < 1 || value > UINT32_MAX || std::trunc(value) != value)
        return StreamStatus::invalid_frame;
    const auto selected = static_cast<TargetId>(value);
    SelectionEvent selection{.time_ns = now,
                             .sequence = result.interval.key,
                             .trial = snapshot.trial,
                             .paradigm = 1,
                             .correct = selected == snapshot.trial.target_id,
                             .selected_id = selected,
                             .intended_id = snapshot.trial.target_id};
    SSVEPStepResult step;
    if (machine_.step(now, selection, step) != ContractStatus::ok)
        return StreamStatus::invalid_state;
    return accept(step);
}

void SSVEPController::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
}
void SSVEPController::stop_execution(bool) noexcept
{
    cancel();
    if (task_thread_.joinable())
        task_thread_.join();
    processing_.stop();
    if (started_ && !machine_.complete())
    {
        SSVEPStepResult result;
        if (machine_.stop(last_time_, result) == ContractStatus::ok)
            static_cast<void>(accept(result));
    }
}
StreamStatus SSVEPController::pop_snapshot(SSVEPSnapshot& snapshot) noexcept
{
    return snapshots_.try_pop(snapshot);
}
std::uint64_t SSVEPController::dropped_trace_count() const noexcept
{
    return traces_.dropped() + snapshots_.dropped() + displays_.dropped();
}
SessionMetadata SSVEPController::metadata() const noexcept
{
    return {.experiment_type = "ssvep",
            .schedule_seed = task_.seed,
            .sampler_version = task_.sampler_version};
}
void SSVEPController::write_configuration(ExperimentTraceSink& sink, ExperimentTimeNs now) noexcept
{
    auto& text = sink.named(ControlKind::task_variables, "ssvep.configuration", now);
    text.u64("calibration_trials", calibration_);
    text.u64("trials", task_.n_trials);
    text.u64("seed", task_.seed);
    text.u64("cue_duration_ns", task_.cue_duration_ns);
    text.u64("stimulation_duration_ns", task_.stimulation_duration_ns);
    text.u64("decision_timeout_ns", task_.decision_timeout_ns);
    text.u64("feedback_duration_ns", task_.feedback_duration_ns);
    text.u64("inter_trial_ns", task_.inter_trial_ns);
    text.u64("window_ns", window_ns_);
    text.u64("shift_ns", shift_ns_);
    static_cast<void>(sink.commit());
    for (std::size_t i = 0; i < feature_names_.size(); ++i)
    {
        auto& feature = sink.named(ControlKind::task_variables, "ssvep.feature_schema", now);
        feature.u64("column", i);
        feature.text("name", feature_names_[i]);
        static_cast<void>(sink.commit());
    }
    for (std::size_t i = 0; i < task_.n_targets; ++i)
    {
        auto& target = sink.named(ControlKind::targets, "ssvep.target", now);
        target.u64("id", task_.targets[i].id);
        target.f64("frequency_hz", task_.targets[i].frequency_hz);
        static_cast<void>(sink.commit());
    }
}
std::size_t SSVEPController::drain(ExperimentTraceSink& sink, std::size_t budget) noexcept
{
    std::size_t count = 0;
    if (!publication_written_ && published_.load(std::memory_order_acquire))
    {
        auto& model = sink.named(ControlKind::task_variables, "ssvep.decoder_publication",
                                 publication_.time_ns);
        model.u64("version", 2);
        model.u64("training_trials", calibration_);
        model.text("fingerprint", publication_.fingerprint);
        model.u64("n_features", n_features_);
        static_cast<void>(sink.commit());
        constexpr std::size_t chunk_size = 256;
        for (std::size_t offset = 0; offset < publication_.plan.size(); offset += chunk_size)
        {
            auto& chunk =
                sink.named(ControlKind::task_variables, "ssvep.decoder_plan", publication_.time_ns);
            chunk.u64("offset", offset);
            chunk.text("json", std::string_view(publication_.plan).substr(offset, chunk_size));
            static_cast<void>(sink.commit());
        }
        publication_written_ = true;
    }
    IntervalFeatureValue feature;
    while (count < budget && processing_.pop_feature(feature) == StreamStatus::ok)
    {
        if (feature.center >= epoch_)
        {
            auto& text = sink.named(ControlKind::labels, "ssvep.feature", feature.center - epoch_);
            text.u64("sample", feature.sample);
            text.u64("column", feature.column);
            text.f64("value", feature.value);
            static_cast<void>(sink.commit());
        }
        ++count;
    }
    SSVEPDisplayEvidence display;
    while (count < budget && displays_.try_pop(display) == StreamStatus::ok)
    {
        auto& text = sink.named(ControlKind::events, "ssvep.presentation", display.requested);
        text.trial("trial", display.snapshot.trial);
        text.u64("state", static_cast<std::uint64_t>(display.snapshot.state));
        text.u64("submitted_host_ns", display.submitted);
        text.u64("presented_host_ns", display.presented);
        text.boolean("expired_snapshot", display.expired);
        static_cast<void>(sink.commit());
        ++count;
    }
    SSVEPStepResult step;
    while (count < budget && traces_.try_pop(step) == StreamStatus::ok)
    {
        ++count;
        for (std::size_t i = 0; i < step.n_transitions; ++i)
        {
            const auto& transition = step.transitions[i];
            auto& text =
                sink.named(ControlKind::experiment_states, "ssvep.transition", transition.time_ns);
            text.u64("sequence", transition.sequence);
            text.u64("from_state", transition.from_state);
            text.u64("to_state", transition.to_state);
            text.u64("cause", transition.cause);
            text.trial("trial", transition.trial);
            static_cast<void>(sink.commit());
        }
        if (step.trial_decided)
        {
            const auto& trial = step.trial;
            const auto ordinal = trial.record.trial.ordinal;
            auto& text = sink.trial(trial.record);
            text.text("selection_source", ordinal < calibration_ ? "calibration_label" : "decoder");
            text.u64("selected_id", trial.has_selection ? trial.selection.selected_id : 0);
            const auto feature_windows = processing_.count(ordinal);
            text.u64("feature_windows", feature_windows);
            const auto interval = processing_.interval(ordinal);
            if (interval.key)
            {
                text.u64("feature_interval_key", interval.key);
                text.u64("feature_start_ns", interval.start_ns - epoch_);
                text.u64("feature_end_ns", interval.end_ns - epoch_);
                text.u64("feature_center_ns",
                         interval.start_ns - epoch_ + (interval.end_ns - interval.start_ns) / 2);
            }
            text.u64("decoder_version", ordinal < calibration_ ? 0 : 2);
            static_cast<void>(sink.commit());
            if (ordinal < ready_trials() && feature_windows != 0)
            {
                for (std::size_t i = 0; i < n_features_; ++i)
                {
                    auto& feature = sink.named(ControlKind::labels, "ssvep.trial_feature",
                                               trial.record.interval.end_ns);
                    feature.u64("trial", ordinal);
                    feature.u64("column", i);
                    feature.f64("value", processing_.features(ordinal)[i]);
                    static_cast<void>(sink.commit());
                }
            }
        }
    }
    return count;
}
void SSVEPController::write_summary(ExperimentTraceSink& sink, ExperimentTimeNs now) noexcept
{
    auto& text = sink.named(ControlKind::task_variables, "ssvep.summary", now);
    text.u64("completed_trials", completed_trials());
    text.boolean("complete", complete());
    text.u64("trace_drops", dropped_trace_count());
    static_cast<void>(sink.commit());
}
} // namespace neurale::execution
