// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "surface.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if defined(_WIN32)
#include <GL/gl.h> // windows.h must precede GL/gl.h for WINGDIAPI and APIENTRY
#elif defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_TEXTURE_2D
#define GL_TEXTURE_2D 0x0DE1
#endif
#ifndef GL_TEXTURE_MIN_FILTER
#define GL_TEXTURE_MIN_FILTER 0x2801
#endif
#ifndef GL_TEXTURE_MAG_FILTER
#define GL_TEXTURE_MAG_FILTER 0x2800
#endif
#ifndef GL_TEXTURE_WRAP_S
#define GL_TEXTURE_WRAP_S 0x2802
#endif
#ifndef GL_TEXTURE_WRAP_T
#define GL_TEXTURE_WRAP_T 0x2803
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_RED
#define GL_RED 0x1903
#endif
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif

namespace neurale::experiment_presentation
{
namespace
{

using GlSize = std::ptrdiff_t;
using GlOffset = std::ptrdiff_t;

struct GlFunctions
{
    void(APIENTRY* gen_vertex_arrays)(GLsizei, GLuint*){};
    void(APIENTRY* bind_vertex_array)(GLuint){};
    void(APIENTRY* delete_vertex_arrays)(GLsizei, const GLuint*){};
    void(APIENTRY* gen_buffers)(GLsizei, GLuint*){};
    void(APIENTRY* bind_buffer)(GLenum, GLuint){};
    void(APIENTRY* buffer_data)(GLenum, GlSize, const void*, GLenum){};
    void(APIENTRY* buffer_sub_data)(GLenum, GlOffset, GlSize, const void*){};
    void(APIENTRY* delete_buffers)(GLsizei, const GLuint*){};
    GLuint(APIENTRY* create_shader)(GLenum) {};
    void(APIENTRY* shader_source)(GLuint, GLsizei, const char* const*, const GLint*){};
    void(APIENTRY* compile_shader)(GLuint){};
    void(APIENTRY* get_shader_iv)(GLuint, GLenum, GLint*){};
    void(APIENTRY* delete_shader)(GLuint){};
    GLuint(APIENTRY* create_program)() {};
    void(APIENTRY* attach_shader)(GLuint, GLuint){};
    void(APIENTRY* link_program)(GLuint){};
    void(APIENTRY* get_program_iv)(GLuint, GLenum, GLint*){};
    void(APIENTRY* delete_program)(GLuint){};
    void(APIENTRY* use_program)(GLuint){};
    void(APIENTRY* enable_vertex_attrib_array)(GLuint){};
    void(APIENTRY* vertex_attrib_pointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*){};
    GLint(APIENTRY* get_uniform_location)(GLuint, const char*) {};
    void(APIENTRY* uniform_1i)(GLint, GLint){};
    void(APIENTRY* active_texture)(GLenum){};
    void(APIENTRY* gen_textures)(GLsizei, GLuint*){};
    void(APIENTRY* bind_texture)(GLenum, GLuint){};
    void(APIENTRY* tex_image_2d)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum,
                                 const void*){};
    void(APIENTRY* tex_parameter_i)(GLenum, GLenum, GLint){};
    void(APIENTRY* delete_textures)(GLsizei, const GLuint*){};
    void(APIENTRY* draw_arrays)(GLenum, GLint, GLsizei){};
    void(APIENTRY* viewport)(GLint, GLint, GLsizei, GLsizei){};
    void(APIENTRY* scissor)(GLint, GLint, GLsizei, GLsizei){};
    void(APIENTRY* clear_color)(GLfloat, GLfloat, GLfloat, GLfloat){};
    void(APIENTRY* clear)(GLbitfield){};
    void(APIENTRY* enable)(GLenum){};
    void(APIENTRY* disable)(GLenum){};
    void(APIENTRY* blend_func)(GLenum, GLenum){};
    void(APIENTRY* pixel_store_i)(GLenum, GLint){};
    GLenum(APIENTRY* get_error)() {};
};

template <class Proc> bool load_gl_proc(Proc& output, const char* name) noexcept
{
    output = reinterpret_cast<Proc>(glfwGetProcAddress(name));
    return output != nullptr;
}

bool load_gl_functions(GlFunctions& gl) noexcept
{
    return load_gl_proc(gl.gen_vertex_arrays, "glGenVertexArrays") &&
           load_gl_proc(gl.bind_vertex_array, "glBindVertexArray") &&
           load_gl_proc(gl.delete_vertex_arrays, "glDeleteVertexArrays") &&
           load_gl_proc(gl.gen_buffers, "glGenBuffers") &&
           load_gl_proc(gl.bind_buffer, "glBindBuffer") &&
           load_gl_proc(gl.buffer_data, "glBufferData") &&
           load_gl_proc(gl.buffer_sub_data, "glBufferSubData") &&
           load_gl_proc(gl.delete_buffers, "glDeleteBuffers") &&
           load_gl_proc(gl.create_shader, "glCreateShader") &&
           load_gl_proc(gl.shader_source, "glShaderSource") &&
           load_gl_proc(gl.compile_shader, "glCompileShader") &&
           load_gl_proc(gl.get_shader_iv, "glGetShaderiv") &&
           load_gl_proc(gl.delete_shader, "glDeleteShader") &&
           load_gl_proc(gl.create_program, "glCreateProgram") &&
           load_gl_proc(gl.attach_shader, "glAttachShader") &&
           load_gl_proc(gl.link_program, "glLinkProgram") &&
           load_gl_proc(gl.get_program_iv, "glGetProgramiv") &&
           load_gl_proc(gl.delete_program, "glDeleteProgram") &&
           load_gl_proc(gl.use_program, "glUseProgram") &&
           load_gl_proc(gl.enable_vertex_attrib_array, "glEnableVertexAttribArray") &&
           load_gl_proc(gl.vertex_attrib_pointer, "glVertexAttribPointer") &&
           load_gl_proc(gl.get_uniform_location, "glGetUniformLocation") &&
           load_gl_proc(gl.uniform_1i, "glUniform1i") &&
           load_gl_proc(gl.active_texture, "glActiveTexture") &&
           load_gl_proc(gl.gen_textures, "glGenTextures") &&
           load_gl_proc(gl.bind_texture, "glBindTexture") &&
           load_gl_proc(gl.tex_image_2d, "glTexImage2D") &&
           load_gl_proc(gl.tex_parameter_i, "glTexParameteri") &&
           load_gl_proc(gl.delete_textures, "glDeleteTextures") &&
           load_gl_proc(gl.draw_arrays, "glDrawArrays") &&
           load_gl_proc(gl.viewport, "glViewport") && load_gl_proc(gl.scissor, "glScissor") &&
           load_gl_proc(gl.clear_color, "glClearColor") && load_gl_proc(gl.clear, "glClear") &&
           load_gl_proc(gl.enable, "glEnable") && load_gl_proc(gl.disable, "glDisable") &&
           load_gl_proc(gl.blend_func, "glBlendFunc") &&
           load_gl_proc(gl.pixel_store_i, "glPixelStorei") &&
           load_gl_proc(gl.get_error, "glGetError");
}

struct Vertex
{
    float x{};
    float y{};
    float red{};
    float green{};
    float blue{};
    float alpha{};
    float u{};
    float v{};
    float textured{};
};

struct DrawBatch
{
    GLenum mode{};
    std::size_t first{};
    std::size_t count{};
};

constexpr const char* kVertexShader = R"glsl(
#version 330 core
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in float in_textured;
out vec4 vertex_color;
out vec2 texture_uv;
flat out float use_texture;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    vertex_color = in_color;
    texture_uv = in_uv;
    use_texture = in_textured;
}
)glsl";

constexpr const char* kFragmentShader = R"glsl(
#version 330 core
in vec4 vertex_color;
in vec2 texture_uv;
flat in float use_texture;
uniform sampler2D glyph_atlas;
out vec4 output_color;
void main() {
    float coverage = use_texture > 0.5 ? texture(glyph_atlas, texture_uv).r : 1.0;
    output_color = vec4(vertex_color.rgb, vertex_color.a * coverage);
}
)glsl";

bool valid_color(Color color) noexcept
{
    return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue) &&
           std::isfinite(color.alpha) && color.red >= 0.0F && color.red <= 1.0F &&
           color.green >= 0.0F && color.green <= 1.0F && color.blue >= 0.0F && color.blue <= 1.0F &&
           color.alpha >= 0.0F && color.alpha <= 1.0F;
}

bool valid_point(Point2d point) noexcept
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

template <std::size_t Size>
void copy_gl_string(std::array<char, Size>& destination, const GLubyte* source) noexcept
{
    destination.fill('\0');
    if (source == nullptr)
        return;
    const auto* text = reinterpret_cast<const char*>(source);
    const auto length = (std::min)(std::strlen(text), Size - 1);
    std::memcpy(destination.data(), text, length);
}

InputAction input_action(int action) noexcept
{
    if (action == GLFW_PRESS)
    {
        return InputAction::press;
    }
    if (action == GLFW_RELEASE)
    {
        return InputAction::release;
    }
    if (action == GLFW_REPEAT)
    {
        return InputAction::repeat;
    }
    return InputAction::none;
}

/// Process-wide GLFW library lifetime, shared by every PresentationSurface.
///
/// glfwInit()/glfwTerminate() are library-global and GLFW keeps no reference
/// count of its own: glfwTerminate() "destroys all remaining windows". A
/// surface that terminated the library on its own close() would therefore
/// destroy every *other* live surface's GLFWwindow* behind its back, leaving
/// those surfaces holding freed pointers that their own close() would then pass
/// to glfwMakeContextCurrent()/glfwDestroyWindow(). Center-Out, WebGrid, and
/// Speech presenters are separately constructible -- from native orchestration
/// and from the Python bindings alike -- so more than one live surface is a
/// supported state, not a misuse to be documented away.
///
/// The counter makes the library outlive the last surface that needs it and no
/// longer. It is guarded by a mutex because the count is the one piece of
/// surface state that is not owned by a single presentation thread; the GLFW
/// calls it guards still happen on the calling (owner) thread, which is the
/// thread GLFW requires for both init and terminate.
class GlfwLibrary
{
  public:
    /// Initialize the library if this is the first user. Returns false only
    /// when glfwInit() itself failed, in which case nothing was acquired.
    [[nodiscard]] static bool acquire() noexcept
    {
        const std::lock_guard<std::mutex> guard(mutex());
        auto& count = use_count();
        if (count == 0 && glfwInit() != GLFW_TRUE)
        {
            return false;
        }
        ++count;
        return true;
    }

    /// Release one use. The library is terminated only by the last user.
    static void release() noexcept
    {
        const std::lock_guard<std::mutex> guard(mutex());
        auto& count = use_count();
        if (count == 0)
        {
            return;
        }
        if (--count == 0)
        {
            glfwTerminate();
        }
    }

    /// Bind @p window's context to the calling thread when it is not already
    /// bound there.
    ///
    /// GLFW's current context is per-thread, so two surfaces owned by one
    /// thread share a single binding slot: without this, the second surface to
    /// open would leave its context current and the first surface's draw calls
    /// would silently land in the wrong context. The thread-local cache keeps
    /// the ordinary single-surface case at zero extra platform calls -- the
    /// binding only changes when the surface being rendered changes.
    static void make_current(GLFWwindow* window) noexcept
    {
        if (current() == window)
        {
            return;
        }
        glfwMakeContextCurrent(window);
        current() = window;
    }

    /// Forget a binding whose window is about to be, or has been, destroyed.
    static void forget_current(GLFWwindow* window) noexcept
    {
        if (current() == window)
        {
            current() = nullptr;
        }
    }

  private:
    [[nodiscard]] static GLFWwindow*& current() noexcept
    {
        static thread_local GLFWwindow* value{};
        return value;
    }
    [[nodiscard]] static std::mutex& mutex() noexcept
    {
        static std::mutex value;
        return value;
    }
    [[nodiscard]] static std::size_t& use_count() noexcept
    {
        static std::size_t value{};
        return value;
    }
};

} // namespace

class PresentationSurface::Impl
{
  public:
    ~Impl()
    {
        close();
    }

    SurfaceStatus open(const WindowConfig& config, RendererTimeNs renderer_origin_ns,
                       experiments::ExperimentTimeNs experiment_origin_ns)
    {
        if (lifecycle_ != SurfaceLifecycle::closed)
        {
            return set_status(SurfaceStatus::invalid_state);
        }
        if (config.window_size.width <= 0 || config.window_size.height <= 0 ||
            config.input_capacity == 0 || (config.swap_interval != 0 && config.swap_interval != 1))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        // See RendererTimeMapper's class documentation for why the mapping is
        // sound only when the renderer origin comes from this clock, and why a
        // future-dated origin is rejected here rather than mapped.
        if (renderer_origin_ns > renderer_monotonic_now_ns())
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        if (failure_ == SurfaceFailurePoint::glfw_initialization)
        {
            failure_ = SurfaceFailurePoint::none;
            return set_status(SurfaceStatus::glfw_initialization_failed);
        }
        if (!GlfwLibrary::acquire())
        {
            return set_status(SurfaceStatus::glfw_initialization_failed);
        }
        glfw_initialized_ = true;
        if (failure_ == SurfaceFailurePoint::window_creation)
        {
            failure_ = SurfaceFailurePoint::none;
            close();
            return set_status(SurfaceStatus::window_creation_failed);
        }

        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
#if defined(__APPLE__)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
        glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, config.visible ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_AUTO_ICONIFY, config.fullscreen ? GLFW_FALSE : GLFW_TRUE);

        GLFWmonitor* monitor = nullptr;
        int monitor_count_value{};
        auto** monitors = glfwGetMonitors(&monitor_count_value);
        n_monitors_ = monitor_count_value;
        if (config.fullscreen || config.monitor_idx >= 0)
        {
            const auto idx = config.monitor_idx >= 0 ? config.monitor_idx : 0;
            if (monitors == nullptr || idx < 0 || idx >= monitor_count_value)
            {
                close();
                return set_status(SurfaceStatus::monitor_unavailable);
            }
            monitor = monitors[idx];
        }

        auto window_size = config.window_size;
        if (config.fullscreen)
        {
            const auto* mode = glfwGetVideoMode(monitor);
            if (mode == nullptr)
            {
                close();
                return set_status(SurfaceStatus::monitor_unavailable);
            }
            window_size = {mode->width, mode->height};
            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        }

        const std::string title(config.title);
        window_ = glfwCreateWindow(window_size.width, window_size.height, title.c_str(),
                                   config.fullscreen ? monitor : nullptr, nullptr);
        if (window_ == nullptr)
        {
            close();
            return set_status(SurfaceStatus::window_creation_failed);
        }
        if (!config.fullscreen && monitor != nullptr)
        {
            int monitor_x{};
            int monitor_y{};
            glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);
            const auto* mode = glfwGetVideoMode(monitor);
            if (mode != nullptr)
            {
                glfwSetWindowPos(window_, monitor_x + (mode->width - config.window_size.width) / 2,
                                 monitor_y + (mode->height - config.window_size.height) / 2);
            }
        }
        GlfwLibrary::make_current(window_);
        glfwSwapInterval(config.swap_interval);
        if (failure_ == SurfaceFailurePoint::opengl_loading)
        {
            failure_ = SurfaceFailurePoint::none;
            close();
            return set_status(SurfaceStatus::opengl_function_missing);
        }
        if (!load_gl_functions(gl_))
        {
            close();
            return set_status(SurfaceStatus::opengl_function_missing);
        }

        environment_ = {};
        copy_gl_string(environment_.gpu_vendor, glGetString(GL_VENDOR));
        copy_gl_string(environment_.gpu_renderer, glGetString(GL_RENDERER));
        copy_gl_string(environment_.opengl_version, glGetString(GL_VERSION));
        environment_.monitor_idx = config.monitor_idx;
        environment_.swap_interval = config.swap_interval;
        environment_.fullscreen = config.fullscreen;
        GLFWmonitor* metadata_monitor = monitor != nullptr ? monitor : glfwGetPrimaryMonitor();
        if (metadata_monitor != nullptr)
        {
            const auto* mode = glfwGetVideoMode(metadata_monitor);
            if (mode != nullptr)
                environment_.refresh_rate_hz = mode->refreshRate;
        }

        int window_width{};
        int window_height{};
        int framebuffer_width{};
        int framebuffer_height{};
        glfwGetWindowSize(window_, &window_width, &window_height);
        glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
        environment_.window_width = window_width;
        environment_.window_height = window_height;
        environment_.framebuffer_width = framebuffer_width;
        environment_.framebuffer_height = framebuffer_height;
        const auto coordinate_status = coordinates_.configure(
            config.logical_space, Size2i{window_width, window_height},
            Size2i{framebuffer_width, framebuffer_height}, config.aspect_policy);
        if (coordinate_status != SurfaceStatus::ok)
        {
            close();
            return set_status(coordinate_status);
        }
        const auto input_status = input_.prepare(config.input_capacity);
        if (input_status != SurfaceStatus::ok)
        {
            close();
            return set_status(input_status);
        }
        time_mapper_.configure(renderer_origin_ns, experiment_origin_ns);
        cancel_requested_.store(false, std::memory_order_release);
        logical_space_ = config.logical_space;
        aspect_policy_ = config.aspect_policy;
        admission_ = config.input_admission;
        owner_thread_ = std::this_thread::get_id();
        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, &Impl::key_callback);
        glfwSetCursorPosCallback(window_, &Impl::cursor_callback);
        glfwSetMouseButtonCallback(window_, &Impl::mouse_button_callback);
        glfwSetWindowCloseCallback(window_, &Impl::close_callback);
        glfwSetWindowSizeCallback(window_, &Impl::window_size_callback);
        glfwSetFramebufferSizeCallback(window_, &Impl::framebuffer_size_callback);
        lifecycle_ = SurfaceLifecycle::open;
        return set_status(SurfaceStatus::ok);
    }

    SurfaceStatus prepare(const SurfacePreparation& config,
                          std::span<const TextCatalogEntry> text_catalog)
    {
        if (!on_owner_thread())
        {
            return set_status(SurfaceStatus::wrong_thread);
        }
        if (lifecycle_ != SurfaceLifecycle::open)
        {
            return set_status(SurfaceStatus::invalid_state);
        }
        if (config.max_vertices == 0 || config.max_draw_batches == 0 ||
            config.circle_segments < 3 || config.circle_segments > config.max_vertices)
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        // Every GL call below belongs to this surface's context, which is not
        // necessarily the one another surface left current on this thread.
        GlfwLibrary::make_current(window_);
        vertices_.reserve(config.max_vertices);
        batches_.reserve(config.max_draw_batches);
        // The GL vertex buffer is sized from max_vertices, so admission has to
        // be judged against max_vertices too. reserve() is only required to
        // give *at least* the requested capacity; gating on capacity() would
        // let an over-allocating allocator admit more vertices than the VBO
        // holds and turn a bounded-resource guarantee into a glBufferSubData
        // overrun caught only by glGetError.
        max_vertices_ = config.max_vertices;
        max_batches_ = config.max_draw_batches;
        circle_segments_ = config.circle_segments;

        if (failure_ == SurfaceFailurePoint::shader_compilation)
        {
            failure_ = SurfaceFailurePoint::none;
            clear_cpu_resources();
            return set_status(SurfaceStatus::shader_compilation_failed);
        }
        const auto shader_status = create_graphics(config.max_vertices);
        if (shader_status != SurfaceStatus::ok)
        {
            destroy_graphics();
            clear_cpu_resources();
            return set_status(shader_status);
        }

        if (!text_catalog.empty())
        {
            if (failure_ == SurfaceFailurePoint::font_preparation)
            {
                failure_ = SurfaceFailurePoint::none;
                destroy_graphics();
                clear_cpu_resources();
                return set_status(SurfaceStatus::font_load_failed);
            }
            const auto text_status = text_.prepare(config.text, text_catalog);
            if (text_status != SurfaceStatus::ok)
            {
                destroy_graphics();
                clear_cpu_resources();
                return set_status(text_status);
            }
            gl_.gen_textures(1, &atlas_texture_);
            gl_.bind_texture(GL_TEXTURE_2D, atlas_texture_);
            gl_.pixel_store_i(GL_UNPACK_ALIGNMENT, 1);
            gl_.tex_image_2d(GL_TEXTURE_2D, 0, GL_RED, text_.width(), text_.height(), 0, GL_RED,
                             GL_UNSIGNED_BYTE, text_.pixels().data());
            gl_.tex_parameter_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            gl_.tex_parameter_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            gl_.tex_parameter_i(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl_.tex_parameter_i(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            if (gl_.get_error() != GL_NO_ERROR)
            {
                destroy_graphics();
                return set_status(SurfaceStatus::presentation_failed);
            }
        }
        lifecycle_ = SurfaceLifecycle::prepared;
        return set_status(SurfaceStatus::ok);
    }

    SurfaceStatus pump_events() noexcept
    {
        if (!on_owner_thread())
        {
            return set_status(SurfaceStatus::wrong_thread);
        }
        if (lifecycle_ != SurfaceLifecycle::open && lifecycle_ != SurfaceLifecycle::prepared)
        {
            return status_;
        }
        if (cancel_requested_.load(std::memory_order_acquire))
        {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
            lifecycle_ = SurfaceLifecycle::cancelled;
            return set_status(SurfaceStatus::cancelled);
        }
        glfwPollEvents();
        if (input_.status() != SurfaceStatus::ok)
        {
            lifecycle_ = SurfaceLifecycle::faulted;
            return set_status(input_.status());
        }
        if (cancel_requested_.load(std::memory_order_acquire))
        {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
            lifecycle_ = SurfaceLifecycle::cancelled;
            return set_status(SurfaceStatus::cancelled);
        }
        if (glfwWindowShouldClose(window_) == GLFW_TRUE)
        {
            lifecycle_ = SurfaceLifecycle::cancelled;
            return set_status(SurfaceStatus::window_closed);
        }
        return set_status(SurfaceStatus::ok);
    }

    SurfaceStatus poll_input(InputEvent& event, bool& available) noexcept
    {
        if (!on_owner_thread())
        {
            available = false;
            return SurfaceStatus::wrong_thread;
        }
        available = input_.pop(event);
        return SurfaceStatus::ok;
    }

    SurfaceStatus begin_frame(Color background) noexcept
    {
        if (!ready_for_render() || !valid_color(background))
        {
            return status_ == SurfaceStatus::ok ? set_status(SurfaceStatus::invalid_configuration)
                                                : status_;
        }
        GlfwLibrary::make_current(window_);
        vertices_.clear();
        batches_.clear();
        // The background clears the whole framebuffer, including any letterbox
        // bars, so the scissor test is off for the clear and the viewport stays
        // the full framebuffer -- the vertex shader receives NDC and
        // framebuffer_to_ndc is expressed against the full framebuffer, so
        // narrowing glViewport would rescale primitives rather than clip them.
        // Logical primitives are then clipped to the letterboxed viewport, so a
        // task primitive outside the logical space (a cursor that left the
        // nominal task bounds, for example) cannot be drawn into a letterbox the
        // input contract already says is outside the task. glScissor is
        // bottom-left origin; Rect2i::viewport() is top-left origin. This is
        // presentation clipping only and never writes a clipped position back to
        // the input mapper, which already excludes the letterbox.
        const auto framebuffer = coordinates_.framebuffer_size();
        const auto viewport = coordinates_.viewport();
        gl_.viewport(0, 0, framebuffer.width, framebuffer.height);
        gl_.disable(GL_SCISSOR_TEST);
        gl_.clear_color(background.red, background.green, background.blue, background.alpha);
        gl_.clear(GL_COLOR_BUFFER_BIT);
        gl_.scissor(viewport.left, framebuffer.height - (viewport.top + viewport.height),
                    viewport.width, viewport.height);
        gl_.enable(GL_SCISSOR_TEST);
        return SurfaceStatus::ok;
    }

    SurfaceStatus line(Point2d start, Point2d end, Color color) noexcept
    {
        if (!valid_point(start) || !valid_point(end) || !valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        const std::array points{start, end};
        return add_untextured(points, color, GL_LINES);
    }

    SurfaceStatus polyline(std::span<const Point2d> points, Color color) noexcept
    {
        if (points.size() < 2 || !valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        for (const auto point : points)
        {
            if (!valid_point(point))
            {
                return set_status(SurfaceStatus::invalid_configuration);
            }
        }
        return add_untextured(points, color, GL_LINE_STRIP);
    }

    SurfaceStatus rectangle(Rect2d rectangle, Color color, bool filled) noexcept
    {
        if (!std::isfinite(rectangle.left) || !std::isfinite(rectangle.bottom) ||
            !std::isfinite(rectangle.width) || !std::isfinite(rectangle.height) ||
            rectangle.width <= 0.0 || rectangle.height <= 0.0 || !valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        const std::array corners{
            Point2d{rectangle.left, rectangle.bottom},
            Point2d{rectangle.left + rectangle.width, rectangle.bottom},
            Point2d{rectangle.left + rectangle.width, rectangle.bottom + rectangle.height},
            Point2d{rectangle.left, rectangle.bottom + rectangle.height},
        };
        if (!filled)
        {
            return add_untextured(corners, color, GL_LINE_LOOP);
        }
        const std::array triangles{
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        };
        return add_untextured(triangles, color, GL_TRIANGLES);
    }

    SurfaceStatus circle(Point2d center, double radius, Color color, bool filled) noexcept
    {
        if (!ready_for_render())
        {
            return status_;
        }
        if (!valid_point(center) || !std::isfinite(radius) || radius <= 0.0 || !valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        const auto required = filled ? circle_segments_ * 3 : circle_segments_;
        if (!can_append(required, 1))
        {
            return set_status(SurfaceStatus::resource_capacity_exceeded);
        }
        const auto first = vertices_.size();
        for (std::size_t segment = 0; segment < circle_segments_; ++segment)
        {
            const auto angle = 2.0 * std::numbers::pi_v<double> * segment / circle_segments_;
            const auto next_angle =
                2.0 * std::numbers::pi_v<double> * (segment + 1) / circle_segments_;
            const auto current = Point2d{
                center.x + radius * std::cos(angle),
                center.y + radius * std::sin(angle),
            };
            if (filled)
            {
                const auto next = Point2d{
                    center.x + radius * std::cos(next_angle),
                    center.y + radius * std::sin(next_angle),
                };
                append_vertex(center, color, false, {});
                append_vertex(current, color, false, {});
                append_vertex(next, color, false, {});
            }
            else
            {
                append_vertex(current, color, false, {});
            }
        }
        batches_.push_back(DrawBatch{
            .mode = static_cast<GLenum>(filled ? GL_TRIANGLES : GL_LINE_LOOP),
            .first = first,
            .count = required,
        });
        return SurfaceStatus::ok;
    }

    SurfaceStatus fixation_cross(Point2d center, double half_extent, Color color) noexcept
    {
        if (!valid_point(center) || !std::isfinite(half_extent) || half_extent <= 0.0)
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        if (!valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        const std::array points{
            Point2d{center.x - half_extent, center.y},
            Point2d{center.x + half_extent, center.y},
            Point2d{center.x, center.y - half_extent},
            Point2d{center.x, center.y + half_extent},
        };
        return add_untextured(points, color, GL_LINES);
    }

    SurfaceStatus shaped_text(PreparedTextId id, Point2d baseline, Color color) noexcept
    {
        if (!ready_for_render() || !text_.prepared())
        {
            return set_status(SurfaceStatus::text_not_prepared);
        }
        if (!valid_point(baseline) || !valid_color(color))
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        const auto* prepared = text_.find(id);
        if (prepared == nullptr)
        {
            ++unexpected_runtime_glyph_cache_misses_;
            return set_status(SurfaceStatus::text_not_prepared);
        }
        auto cursor = coordinates_.logical_to_framebuffer(baseline);
        for (const auto& shaped : prepared->glyphs)
        {
            const auto& glyph = text_.glyph(shaped.atlas_entry);
            if (glyph.width > 0 && glyph.height > 0)
            {
                if (!can_append(6, 1))
                {
                    return set_status(SurfaceStatus::resource_capacity_exceeded);
                }
                const auto left = cursor.x + shaped.x_offset + glyph.bitmap_left;
                const auto top = cursor.y - shaped.y_offset - glyph.bitmap_top;
                const auto right = left + glyph.width;
                const auto bottom = top + glyph.height;
                const auto u0 = static_cast<double>(glyph.atlas_x) / text_.width();
                const auto v0 = static_cast<double>(glyph.atlas_y) / text_.height();
                const auto u1 = static_cast<double>(glyph.atlas_x + glyph.width) / text_.width();
                const auto v1 = static_cast<double>(glyph.atlas_y + glyph.height) / text_.height();
                const auto first = vertices_.size();
                append_framebuffer_vertex({left, top}, color, true, {u0, v0});
                append_framebuffer_vertex({right, top}, color, true, {u1, v0});
                append_framebuffer_vertex({right, bottom}, color, true, {u1, v1});
                append_framebuffer_vertex({left, top}, color, true, {u0, v0});
                append_framebuffer_vertex({right, bottom}, color, true, {u1, v1});
                append_framebuffer_vertex({left, bottom}, color, true, {u0, v1});
                batches_.push_back(DrawBatch{GL_TRIANGLES, first, 6});
            }
            cursor.x += shaped.x_advance;
            cursor.y -= shaped.y_advance;
        }
        return SurfaceStatus::ok;
    }

    SoftwarePresentationTimes present(experiments::ExperimentTimeNs requested_ns,
                                      experiments::ExperimentTimeNs intended_ns) noexcept
    {
        SoftwarePresentationTimes result{
            .status = status_,
            .requested_ns = requested_ns,
            .intended_ns = intended_ns,
        };
        if (!ready_for_render())
        {
            return result;
        }
        if (failure_ == SurfaceFailurePoint::presentation)
        {
            failure_ = SurfaceFailurePoint::none;
            lifecycle_ = SurfaceLifecycle::faulted;
            result.status = set_status(SurfaceStatus::presentation_failed);
            return result;
        }

        result.submitted_renderer_ns = renderer_monotonic_now_ns();
        auto mapping_status = time_mapper_.map(result.submitted_renderer_ns, result.submitted_ns);
        if (mapping_status != SurfaceStatus::ok)
        {
            lifecycle_ = SurfaceLifecycle::faulted;
            result.status = set_status(mapping_status);
            return result;
        }
        gl_.bind_vertex_array(vao_);
        gl_.bind_buffer(GL_ARRAY_BUFFER, vbo_);
        gl_.buffer_sub_data(GL_ARRAY_BUFFER, 0,
                            static_cast<GlSize>(vertices_.size() * sizeof(Vertex)),
                            vertices_.data());
        gl_.use_program(program_);
        gl_.active_texture(GL_TEXTURE0);
        gl_.bind_texture(GL_TEXTURE_2D, atlas_texture_);
        for (const auto& batch : batches_)
        {
            gl_.draw_arrays(batch.mode, static_cast<GLint>(batch.first),
                            static_cast<GLsizei>(batch.count));
        }
        if (gl_.get_error() != GL_NO_ERROR)
        {
            lifecycle_ = SurfaceLifecycle::faulted;
            result.status = set_status(SurfaceStatus::presentation_failed);
            return result;
        }
        glfwSwapBuffers(window_);
        result.presented_renderer_ns = renderer_monotonic_now_ns();
        mapping_status = time_mapper_.map(result.presented_renderer_ns, result.presented_ns);
        if (mapping_status != SurfaceStatus::ok)
        {
            lifecycle_ = SurfaceLifecycle::faulted;
            result.status = set_status(mapping_status);
            return result;
        }
        result.status = set_status(SurfaceStatus::ok);
        return result;
    }

    SurfaceStatus request_window_size(Size2i size) noexcept
    {
        if (!on_owner_thread())
        {
            return set_status(SurfaceStatus::wrong_thread);
        }
        if (window_ == nullptr)
        {
            return set_status(SurfaceStatus::invalid_state);
        }
        if (size.width <= 0 || size.height <= 0)
        {
            return set_status(SurfaceStatus::invalid_configuration);
        }
        glfwSetWindowSize(window_, size.width, size.height);
        return SurfaceStatus::ok;
    }

    int monitor_count() const noexcept
    {
        return n_monitors_;
    }

    void cancel() noexcept
    {
        // The only cross-thread operation. It records a request and touches
        // nothing else -- not window_, not GLFW, not lifecycle_ or status_,
        // which the presentation thread owns. The owner thread observes the
        // request in pump_events() and performs glfwSetWindowShouldClose there.
        // Reading window_ or calling GLFW here would race owner-thread close(),
        // which may glfwTerminate() concurrently; glfwPostEmptyEvent() after
        // glfwTerminate() is undefined. glfwPollEvents() is in use, so there is
        // no blocking event loop to wake -- a future move to glfwWaitEvents()
        // would need a separate, designed wake/lifetime handshake rather than a
        // GLFW call made from this thread.
        cancel_requested_.store(true, std::memory_order_release);
    }

    void close() noexcept
    {
        // close() is owner-thread work: it makes the GLFW context current,
        // deletes GL resources, destroys the window, and terminates GLFW, none
        // of which is safe off the thread that owns the GLFW window. A surface
        // that was never opened, or is already closed, has owner_thread_ reset
        // and no live GLFW state, so close() from any thread is a harmless
        // no-op then. A live surface closed from the wrong thread is a contract
        // violation: report wrong_thread and leave teardown to the owner rather
        // than calling GLFW off-thread. The destructor routes through here too,
        // so wrong-thread destruction is rejected the same way instead of
        // calling GLFW from whatever thread the object was destroyed on.
        if (!on_owner_thread() && (window_ != nullptr || glfw_initialized_))
        {
            set_status(SurfaceStatus::wrong_thread);
            return;
        }
        if (window_ != nullptr)
        {
            GlfwLibrary::make_current(window_);
            destroy_graphics();
            glfwSetWindowUserPointer(window_, nullptr);
            glfwDestroyWindow(window_);
            // The binding cache must not outlive the window it names, or the
            // next surface to render on this thread would skip its own
            // make_current() because a freed pointer compared equal.
            GlfwLibrary::forget_current(window_);
            window_ = nullptr;
        }
        else
        {
            clear_cpu_resources();
        }
        if (glfw_initialized_)
        {
            // Release this surface's use of the library, not the library. Only
            // the last live surface actually terminates GLFW, so closing one
            // surface can no longer destroy another one's window.
            GlfwLibrary::release();
            glfw_initialized_ = false;
        }
        input_.close();
        time_mapper_.reset();
        cancel_requested_.store(false, std::memory_order_release);
        coordinates_ = CoordinateMapper{};
        environment_ = {};
        n_monitors_ = 0;
        unexpected_runtime_glyph_cache_misses_ = 0;
        lifecycle_ = SurfaceLifecycle::closed;
        status_ = SurfaceStatus::ok;
        owner_thread_ = std::thread::id{};
    }

    SurfaceStatus status() const noexcept
    {
        return status_;
    }
    SurfaceLifecycle lifecycle() const noexcept
    {
        return lifecycle_;
    }
    const CoordinateMapper& coordinates() const noexcept
    {
        return coordinates_;
    }

    ResourceStats resource_stats() const noexcept
    {
        const auto text_stats = text_.preparation_stats();
        return ResourceStats{
            .window_open = window_ != nullptr,
            .opengl_ready = vao_ != 0 && vbo_ != 0 && program_ != 0,
            .text_ready = text_.prepared() && atlas_texture_ != 0,
            // The prepared budgets, which are the bound actually enforced by
            // can_append() and the size the GL buffer was built from -- not
            // vector capacity, which is an allocator detail.
            .vertex_capacity = max_vertices_,
            .batch_capacity = max_batches_,
            .input_capacity = input_.capacity(),
            .prepared_text_count = text_.text_count(),
            .prepared_glyph_count = text_.glyph_count(),
            .text_preparation_ns = text_stats.total_ns,
            .shaping_preparation_ns = text_stats.shaping_ns,
            .glyph_cache_preparation_ns = text_stats.glyph_cache_ns,
            .unexpected_runtime_glyph_cache_misses = unexpected_runtime_glyph_cache_misses_,
        };
    }

    PresentationEnvironment environment() const noexcept
    {
        return environment_;
    }

    void inject_failure(SurfaceFailurePoint point) noexcept
    {
        failure_ = point;
    }

    // Test seam: feed an event through enqueue() as a GLFW callback would,
    // without a real window. Going through enqueue() -- not input_.push() --
    // means the test seam exercises the same admission policy a real cursor/key
    // callback does, so a test can prove that admission drops a pointer burst
    // before the FixedInputQueue ever fills.
    void inject_input(const PendingInputEvent& event) noexcept
    {
        enqueue(event);
    }

  private:
    SurfaceStatus set_status(SurfaceStatus status) noexcept
    {
        status_ = status;
        return status;
    }

    bool on_owner_thread() const noexcept
    {
        return owner_thread_ == std::thread::id{} || owner_thread_ == std::this_thread::get_id();
    }

    bool ready_for_render() noexcept
    {
        if (!on_owner_thread())
        {
            set_status(SurfaceStatus::wrong_thread);
            return false;
        }
        if (lifecycle_ != SurfaceLifecycle::prepared || status_ != SurfaceStatus::ok)
        {
            return false;
        }
        return true;
    }

    bool can_append(std::size_t n_vertices, std::size_t n_batches) const noexcept
    {
        // Judged against the prepared budgets, which are also the sizes the GL
        // vertex buffer and the batch list were built from -- never against
        // vector capacity, which may legally exceed what was reserved.
        return n_vertices <= max_vertices_ - vertices_.size() &&
               n_batches <= max_batches_ - batches_.size();
    }

    void append_framebuffer_vertex(Point2d framebuffer, Color color, bool textured,
                                   Point2d uv) noexcept
    {
        const auto ndc = coordinates_.framebuffer_to_ndc(framebuffer);
        vertices_.push_back(Vertex{
            .x = static_cast<float>(ndc.x),
            .y = static_cast<float>(ndc.y),
            .red = color.red,
            .green = color.green,
            .blue = color.blue,
            .alpha = color.alpha,
            .u = static_cast<float>(uv.x),
            .v = static_cast<float>(uv.y),
            .textured = textured ? 1.0F : 0.0F,
        });
    }

    void append_vertex(Point2d logical, Color color, bool textured, Point2d uv) noexcept
    {
        append_framebuffer_vertex(coordinates_.logical_to_framebuffer(logical), color, textured,
                                  uv);
    }

    SurfaceStatus add_untextured(std::span<const Point2d> points, Color color, GLenum mode) noexcept
    {
        if (!ready_for_render())
        {
            return status_;
        }
        if (!can_append(points.size(), 1))
        {
            return set_status(SurfaceStatus::resource_capacity_exceeded);
        }
        const auto first = vertices_.size();
        for (const auto point : points)
        {
            append_vertex(point, color, false, {});
        }
        batches_.push_back(DrawBatch{mode, first, points.size()});
        return SurfaceStatus::ok;
    }

    GLuint compile_shader(GLenum type, const char* source, SurfaceStatus& status) noexcept
    {
        const auto shader = gl_.create_shader(type);
        gl_.shader_source(shader, 1, &source, nullptr);
        gl_.compile_shader(shader);
        GLint compiled{};
        gl_.get_shader_iv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != GL_TRUE)
        {
            gl_.delete_shader(shader);
            status = SurfaceStatus::shader_compilation_failed;
            return 0;
        }
        return shader;
    }

    SurfaceStatus create_graphics(std::size_t max_vertices) noexcept
    {
        SurfaceStatus shader_status{SurfaceStatus::ok};
        const auto vertex_shader = compile_shader(GL_VERTEX_SHADER, kVertexShader, shader_status);
        if (vertex_shader == 0)
        {
            return shader_status;
        }
        const auto fragment_shader =
            compile_shader(GL_FRAGMENT_SHADER, kFragmentShader, shader_status);
        if (fragment_shader == 0)
        {
            gl_.delete_shader(vertex_shader);
            return shader_status;
        }
        program_ = gl_.create_program();
        gl_.attach_shader(program_, vertex_shader);
        gl_.attach_shader(program_, fragment_shader);
        gl_.link_program(program_);
        gl_.delete_shader(vertex_shader);
        gl_.delete_shader(fragment_shader);
        GLint linked{};
        gl_.get_program_iv(program_, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            gl_.delete_program(program_);
            program_ = 0;
            return SurfaceStatus::shader_link_failed;
        }

        gl_.gen_vertex_arrays(1, &vao_);
        gl_.gen_buffers(1, &vbo_);
        gl_.bind_vertex_array(vao_);
        gl_.bind_buffer(GL_ARRAY_BUFFER, vbo_);
        gl_.buffer_data(GL_ARRAY_BUFFER, static_cast<GlSize>(max_vertices * sizeof(Vertex)),
                        nullptr, GL_DYNAMIC_DRAW);
        gl_.enable_vertex_attrib_array(0);
        gl_.vertex_attrib_pointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, x)));
        gl_.enable_vertex_attrib_array(1);
        gl_.vertex_attrib_pointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, red)));
        gl_.enable_vertex_attrib_array(2);
        gl_.vertex_attrib_pointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, u)));
        gl_.enable_vertex_attrib_array(3);
        gl_.vertex_attrib_pointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, textured)));
        gl_.use_program(program_);
        const auto sampler = gl_.get_uniform_location(program_, "glyph_atlas");
        if (sampler >= 0)
        {
            gl_.uniform_1i(sampler, 0);
        }
        gl_.enable(GL_BLEND);
        gl_.enable(GL_MULTISAMPLE);
        gl_.blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        if (gl_.get_error() != GL_NO_ERROR)
        {
            return SurfaceStatus::presentation_failed;
        }
        return SurfaceStatus::ok;
    }

    void destroy_graphics() noexcept
    {
        if (atlas_texture_ != 0 && gl_.delete_textures != nullptr)
        {
            gl_.delete_textures(1, &atlas_texture_);
            atlas_texture_ = 0;
        }
        if (vbo_ != 0 && gl_.delete_buffers != nullptr)
        {
            gl_.delete_buffers(1, &vbo_);
            vbo_ = 0;
        }
        if (vao_ != 0 && gl_.delete_vertex_arrays != nullptr)
        {
            gl_.delete_vertex_arrays(1, &vao_);
            vao_ = 0;
        }
        if (program_ != 0 && gl_.delete_program != nullptr)
        {
            gl_.delete_program(program_);
            program_ = 0;
        }
        clear_cpu_resources();
    }

    void clear_cpu_resources() noexcept
    {
        text_.close();
        vertices_.clear();
        batches_.clear();
        vertices_.shrink_to_fit();
        batches_.shrink_to_fit();
        max_vertices_ = 0;
        max_batches_ = 0;
        circle_segments_ = 0;
    }

    void enqueue(PendingInputEvent event) noexcept
    {
        if (input_.status() != SurfaceStatus::ok)
        {
            return;
        }
        // See InputAdmission's documentation for why this filter runs here,
        // before the event reaches the FixedInputQueue.
        if (!admits(admission_, event.kind))
        {
            return;
        }
        // Only Escape presses enter the ring; ignore releases, repeats and other keys.
        if (event.kind == InputEventKind::key &&
            (event.action != InputAction::press || event.code != GLFW_KEY_ESCAPE))
            return;
        const auto event_status = input_.push(event, time_mapper_);
        if (event_status != SurfaceStatus::ok)
        {
            status_ = event_status;
            lifecycle_ = SurfaceLifecycle::faulted;
        }
    }

    Point2d pointer_position(bool& inside) const noexcept
    {
        double window_x{};
        double window_y{};
        glfwGetCursorPos(window_, &window_x, &window_y);
        Point2d logical{};
        if (!coordinates_.window_pointer_to_logical_unclipped({window_x, window_y}, logical,
                                                              inside))
        {
            inside = false;
            logical = {};
        }
        return logical;
    }

    void refresh_coordinates() noexcept
    {
        int window_width{};
        int window_height{};
        int framebuffer_width{};
        int framebuffer_height{};
        glfwGetWindowSize(window_, &window_width, &window_height);
        glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
        if (window_width > 0 && window_height > 0 && framebuffer_width > 0 &&
            framebuffer_height > 0)
        {
            const auto ignored =
                coordinates_.configure(logical_space_, {window_width, window_height},
                                       {framebuffer_width, framebuffer_height}, aspect_policy_);
            static_cast<void>(ignored);
        }
    }

    static Impl* self(GLFWwindow* window) noexcept
    {
        return static_cast<Impl*>(glfwGetWindowUserPointer(window));
    }

    static void key_callback(GLFWwindow* window, int key, int, int action, int modifiers) noexcept
    {
        if (auto* surface = self(window); surface != nullptr)
        {
            surface->enqueue(PendingInputEvent{
                .renderer_time_ns = renderer_monotonic_now_ns(),
                .kind = InputEventKind::key,
                .action = input_action(action),
                .code = key,
                .modifiers = modifiers,
            });
        }
    }

    static void cursor_callback(GLFWwindow* window, double x, double y) noexcept
    {
        if (auto* surface = self(window); surface != nullptr)
        {
            Point2d logical{};
            bool inside{};
            if (!surface->coordinates_.window_pointer_to_logical_unclipped({x, y}, logical, inside))
            {
                inside = false;
                logical = {};
            }
            surface->enqueue(PendingInputEvent{
                .renderer_time_ns = renderer_monotonic_now_ns(),
                .kind = InputEventKind::pointer_moved,
                .pointer = logical,
                .pointer_inside = inside,
            });
        }
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action,
                                      int modifiers) noexcept
    {
        if (auto* surface = self(window); surface != nullptr)
        {
            bool inside{};
            const auto pointer = surface->pointer_position(inside);
            surface->enqueue(PendingInputEvent{
                .renderer_time_ns = renderer_monotonic_now_ns(),
                .kind = InputEventKind::mouse_button,
                .action = input_action(action),
                .code = button,
                .modifiers = modifiers,
                .pointer = pointer,
                .pointer_inside = inside,
            });
        }
    }

    static void close_callback(GLFWwindow* window) noexcept
    {
        if (auto* surface = self(window); surface != nullptr)
        {
            surface->enqueue(PendingInputEvent{
                .renderer_time_ns = renderer_monotonic_now_ns(),
                .kind = InputEventKind::window_close,
            });
        }
    }

    static void window_size_callback(GLFWwindow* window, int, int) noexcept
    {
        if (auto* surface = self(window); surface != nullptr)
        {
            surface->refresh_coordinates();
        }
    }

    static void framebuffer_size_callback(GLFWwindow* window, int, int) noexcept
    {
        // Resize is a coordinate-mapper refresh, not orchestration evidence: no
        // concrete presenter consumes framebuffer_resized from the queue, so it
        // does not occupy a ring slot. The callback only refreshes the mapper;
        // poll_input/poll_control never see a resize event.
        if (auto* surface = self(window); surface != nullptr)
        {
            surface->refresh_coordinates();
        }
    }

    GLFWwindow* window_{};
    GlFunctions gl_{};
    CoordinateMapper coordinates_{};
    RendererTimeMapper time_mapper_{};
    FixedInputQueue input_{};
    PreparedTextAtlas text_{};
    PresentationEnvironment environment_{};
    std::vector<Vertex> vertices_{};
    std::vector<DrawBatch> batches_{};
    Rect2d logical_space_{};
    AspectPolicy aspect_policy_{AspectPolicy::fit_letterbox};
    InputAdmission admission_{InputAdmission::all};
    std::thread::id owner_thread_{};
    GLuint vao_{};
    GLuint vbo_{};
    GLuint program_{};
    GLuint atlas_texture_{};
    std::size_t max_vertices_{};
    std::size_t max_batches_{};
    std::size_t circle_segments_{};
    int n_monitors_{};
    std::uint64_t unexpected_runtime_glyph_cache_misses_{};
    SurfaceStatus status_{SurfaceStatus::ok};
    SurfaceLifecycle lifecycle_{SurfaceLifecycle::closed};
    SurfaceFailurePoint failure_{SurfaceFailurePoint::none};
    std::atomic<bool> cancel_requested_{};
    bool glfw_initialized_{};
};

PresentationSurface::PresentationSurface() : impl_(std::make_unique<Impl>()) {}
PresentationSurface::~PresentationSurface() = default;

SurfaceStatus PresentationSurface::open(const WindowConfig& config,
                                        RendererTimeNs renderer_origin_ns,
                                        experiments::ExperimentTimeNs experiment_origin_ns)
{
    return impl_->open(config, renderer_origin_ns, experiment_origin_ns);
}

SurfaceStatus PresentationSurface::prepare(const SurfacePreparation& config,
                                           std::span<const TextCatalogEntry> text_catalog)
{
    return impl_->prepare(config, text_catalog);
}

SurfaceStatus PresentationSurface::pump_events() noexcept
{
    return impl_->pump_events();
}
SurfaceStatus PresentationSurface::poll_input(InputEvent& event, bool& available) noexcept
{
    return impl_->poll_input(event, available);
}
SurfaceStatus PresentationSurface::begin_frame(Color background) noexcept
{
    return impl_->begin_frame(background);
}
SurfaceStatus PresentationSurface::line(Point2d start, Point2d end, Color color) noexcept
{
    return impl_->line(start, end, color);
}
SurfaceStatus PresentationSurface::polyline(std::span<const Point2d> points, Color color) noexcept
{
    return impl_->polyline(points, color);
}
SurfaceStatus PresentationSurface::rectangle(Rect2d rectangle, Color color, bool filled) noexcept
{
    return impl_->rectangle(rectangle, color, filled);
}
SurfaceStatus PresentationSurface::circle(Point2d center, double radius, Color color,
                                          bool filled) noexcept
{
    return impl_->circle(center, radius, color, filled);
}
SurfaceStatus PresentationSurface::fixation_cross(Point2d center, double half_extent,
                                                  Color color) noexcept
{
    return impl_->fixation_cross(center, half_extent, color);
}
SurfaceStatus PresentationSurface::shaped_text(PreparedTextId text, Point2d baseline,
                                               Color color) noexcept
{
    return impl_->shaped_text(text, baseline, color);
}
SoftwarePresentationTimes
PresentationSurface::present(experiments::ExperimentTimeNs requested_ns,
                             experiments::ExperimentTimeNs intended_ns) noexcept
{
    return impl_->present(requested_ns, intended_ns);
}
SurfaceStatus PresentationSurface::request_window_size(Size2i size) noexcept
{
    return impl_->request_window_size(size);
}
int PresentationSurface::monitor_count() const noexcept
{
    return impl_->monitor_count();
}
void PresentationSurface::cancel() noexcept
{
    impl_->cancel();
}
void PresentationSurface::close() noexcept
{
    impl_->close();
}
SurfaceStatus PresentationSurface::status() const noexcept
{
    return impl_->status();
}
SurfaceLifecycle PresentationSurface::lifecycle() const noexcept
{
    return impl_->lifecycle();
}
const CoordinateMapper& PresentationSurface::coordinates() const noexcept
{
    return impl_->coordinates();
}
ResourceStats PresentationSurface::resource_stats() const noexcept
{
    return impl_->resource_stats();
}
PresentationEnvironment PresentationSurface::environment() const noexcept
{
    return impl_->environment();
}
void PresentationSurface::inject_failure(SurfaceFailurePoint point) noexcept
{
    impl_->inject_failure(point);
}

void PresentationSurface::inject_input(const PendingInputEvent& event) noexcept
{
    impl_->inject_input(event);
}

} // namespace neurale::experiment_presentation
