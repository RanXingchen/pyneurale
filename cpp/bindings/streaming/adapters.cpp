/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "adapters.h"
#include "types.h"

#include <neurale/streaming/consumer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/source.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace
{

using namespace neurale::streaming;

[[nodiscard]] py::object require_method(const py::object& component, const char* name)
{
    if (!py::hasattr(component, name))
    {
        throw std::invalid_argument(std::string{name} + "() is required");
    }
    auto method = component.attr(name);
    if (!PyCallable_Check(method.ptr()))
    {
        throw std::invalid_argument(std::string{name} + " must be callable");
    }
    return method;
}

[[nodiscard]] py::object optional_method(const py::object& component, const char* name)
{
    if (!py::hasattr(component, name))
    {
        return py::none();
    }
    auto method = component.attr(name);
    if (PyCallable_Check(method.ptr()))
    {
        return method;
    }
    return py::none();
}

template <typename Callback>
StreamStatus invoke_python(Callback&& callback, StreamStatus failure,
                           std::atomic<std::uint64_t>& errors) noexcept
{
    py::gil_scoped_acquire acquire;
    try
    {
        callback();
        return StreamStatus::ok;
    }
    catch (const py::error_already_set&)
    {
        errors.fetch_add(1, std::memory_order_relaxed);
        return failure;
    }
    catch (...)
    {
        errors.fetch_add(1, std::memory_order_relaxed);
        return failure;
    }
}

class PythonSourceAdapter final : public NativeFrameSource
{
  public:
    explicit PythonSourceAdapter(py::object component)
        : component_(std::move(component)), read_(require_method(component_, "read")),
          cancel_(optional_method(component_, "cancel")),
          reset_(optional_method(component_, "reset"))
    {
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        StreamStatus result = StreamStatus::source_failure;
        const auto status = invoke_python(
            [&]
            {
                auto value = read_();
                if (value.is_none())
                {
                    result = StreamStatus::end_of_stream;
                    return;
                }
                result = python::copy_frame(value.cast<const python::Frame&>(), frame);
            },
            StreamStatus::source_failure, callback_errors_);
        return status == StreamStatus::ok ? result : status;
    }

    void cancel() noexcept override
    {
        if (!cancel_.is_none())
        {
            static_cast<void>(
                invoke_python([&] { cancel_(); }, StreamStatus::source_failure, callback_errors_));
        }
    }

    StreamStatus reset() noexcept override
    {
        return reset_.is_none() ? StreamStatus::ok
                                : invoke_python([&] { reset_(); }, StreamStatus::source_failure,
                                                callback_errors_);
    }

    [[nodiscard]] std::uint64_t callback_errors() const noexcept
    {
        return callback_errors_.load(std::memory_order_relaxed);
    }

  private:
    py::object component_;
    py::object read_;
    py::object cancel_;
    py::object reset_;
    std::atomic<std::uint64_t> callback_errors_{};
};

class PythonProcessorAdapter final : public NativeFrameProcessor
{
  public:
    explicit PythonProcessorAdapter(py::object component)
        : component_(std::move(component)), process_(require_method(component_, "process")),
          discontinuity_(optional_method(component_, "handle_discontinuity")),
          flush_(optional_method(component_, "flush")), reset_(optional_method(component_, "reset"))
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {context.input_schema.clone(),
                context.input_schema.clone(),
                context.max_process_outputs,
                context.max_flush_outputs,
                false,
                ProcessorResourceBounds{.frame_pool_leases = 2}};
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& output) noexcept override
    {
        return invoke_outputs([&] { return process_(python::copy_frame(frame.view())); }, output);
    }

    StreamStatus
    handle_discontinuity(const neurale::streaming::Discontinuity& value) noexcept override
    {
        return discontinuity_.is_none()
                   ? StreamStatus::ok
                   : invoke_python([&] { discontinuity_(python::copy_discontinuity(value)); },
                                   StreamStatus::processor_failure, callback_errors_);
    }

    StreamStatus flush(FrameEmitter& output) noexcept override
    {
        return flush_.is_none() ? StreamStatus::ok
                                : invoke_outputs([&] { return flush_(); }, output);
    }

    StreamStatus reset() noexcept override
    {
        return reset_.is_none() ? StreamStatus::ok
                                : invoke_python([&] { reset_(); }, StreamStatus::processor_failure,
                                                callback_errors_);
    }

    [[nodiscard]] std::uint64_t callback_errors() const noexcept
    {
        return callback_errors_.load(std::memory_order_relaxed);
    }

  private:
    template <typename Callback>
    StreamStatus invoke_outputs(Callback&& callback, FrameEmitter& output) noexcept
    {
        StreamStatus result = StreamStatus::processor_failure;
        const auto status = invoke_python(
            [&]
            {
                auto value = callback();
                if (value.is_none())
                {
                    result = StreamStatus::ok;
                    return;
                }
                if (py::isinstance<python::Frame>(value))
                {
                    result = publish(value.template cast<const python::Frame&>(), output);
                    return;
                }
                result = StreamStatus::ok;
                for (const auto item : py::reinterpret_borrow<py::iterable>(value))
                {
                    result = publish(item.template cast<const python::Frame&>(), output);
                    if (result != StreamStatus::ok)
                    {
                        break;
                    }
                }
            },
            StreamStatus::processor_failure, callback_errors_);
        return status == StreamStatus::ok ? result : status;
    }

    static StreamStatus publish(const python::Frame& source, FrameEmitter& output)
    {
        FrameBorrow destination{};
        auto status = output.try_acquire(destination);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        static_cast<void>(python::copy_frame(source, *destination));
        return output.publish_acquired();
    }

    py::object component_;
    py::object process_;
    py::object discontinuity_;
    py::object flush_;
    py::object reset_;
    std::atomic<std::uint64_t> callback_errors_{};
};

class PythonSinkAdapter final : public NativeFrameConsumer
{
  public:
    explicit PythonSinkAdapter(py::object component)
        : component_(std::move(component)), write_(require_method(component_, "write")),
          discontinuity_(optional_method(component_, "handle_discontinuity")),
          flush_(optional_method(component_, "flush")),
          reset_(optional_method(component_, "reset")),
          cancel_(optional_method(component_, "cancel"))
    {
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        return invoke_python([&] { write_(python::copy_frame(frame)); },
                             StreamStatus::consumer_failure, callback_errors_);
    }

    StreamStatus
    handle_discontinuity(const neurale::streaming::Discontinuity& value) noexcept override
    {
        return discontinuity_.is_none()
                   ? StreamStatus::ok
                   : invoke_python([&] { discontinuity_(python::copy_discontinuity(value)); },
                                   StreamStatus::consumer_failure, callback_errors_);
    }

    StreamStatus flush() noexcept override
    {
        return flush_.is_none() ? StreamStatus::ok
                                : invoke_python([&] { flush_(); }, StreamStatus::consumer_failure,
                                                callback_errors_);
    }

    StreamStatus reset() noexcept override
    {
        return reset_.is_none() ? StreamStatus::ok
                                : invoke_python([&] { reset_(); }, StreamStatus::consumer_failure,
                                                callback_errors_);
    }

    void cancel() noexcept override
    {
        if (!cancel_.is_none())
        {
            static_cast<void>(invoke_python([&] { cancel_(); }, StreamStatus::consumer_failure,
                                            callback_errors_));
        }
    }

    [[nodiscard]] std::uint64_t callback_errors() const noexcept
    {
        return callback_errors_.load(std::memory_order_relaxed);
    }

  private:
    py::object component_;
    py::object write_;
    py::object discontinuity_;
    py::object flush_;
    py::object reset_;
    py::object cancel_;
    std::atomic<std::uint64_t> callback_errors_{};
};

} // namespace

void bind_streaming_python_adapters(py::module_& module)
{
    py::class_<PythonSourceAdapter, NativeFrameSource>(module, "PythonSourceAdapter",
                                                       py::is_final())
        .def(py::init<py::object>(), py::arg("component"))
        .def_property_readonly("callback_errors", &PythonSourceAdapter::callback_errors);
    py::class_<PythonProcessorAdapter, NativeFrameProcessor>(module, "PythonProcessorAdapter",
                                                             py::is_final())
        .def(py::init<py::object>(), py::arg("component"))
        .def_property_readonly("callback_errors", &PythonProcessorAdapter::callback_errors);
    py::class_<PythonSinkAdapter, NativeFrameConsumer>(module, "PythonSinkAdapter", py::is_final())
        .def(py::init<py::object>(), py::arg("component"))
        .def_property_readonly("callback_errors", &PythonSinkAdapter::callback_errors);
}
