/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace neurale::streaming
{
class NativeCriticalObserver;
}

namespace neurale::recording
{

class NativeRecorderCore;
class SpoolFile;
class UnixClock;
struct NativeRecordingPlan;

inline constexpr const char* kExperimentRecorderAttachmentCapsule =
    "neurale.recording.experiment_attachment.v1";

/// Non-owning cross-extension view. The Python experiment session retains the
/// recorder owner until native shutdown and offline finalization complete.
struct ExperimentRecorderAttachment
{
    NativeRecorderCore* recorder{};
    const NativeRecordingPlan* plan{};
    SpoolFile* spool{};
    UnixClock* clock{};
    streaming::NativeCriticalObserver* data_observer{};
};

} // namespace neurale::recording
