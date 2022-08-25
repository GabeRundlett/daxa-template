#include "window.hpp"
#include <thread>
#include <iostream>

#include <daxa/utils/imgui.hpp>
#include "imgui/imgui_impl_glfw.h"

using namespace daxa::types;
using Clock = std::chrono::high_resolution_clock;

struct ComputeInput
{
    u32vec2 frame_dim;
    f32vec2 view_origin = {0.0f, 0.0f};
    f32 zoom = 2.0f;
    f32 time;
    i32 max_steps = 512;
};

struct ComputePush
{
    daxa::ImageViewId image_id;
    daxa::BufferId input_buffer_id;
};

struct App : AppWindow<App>
{
    daxa::Context daxa_ctx = daxa::create_context({
        .enable_validation = true,
    });
    daxa::Device device = daxa_ctx.create_device({});

    daxa::Swapchain swapchain = device.create_swapchain({
        .native_window = get_native_handle(),
        .width = size_x,
        .height = size_y,
        .surface_format_selector = [](daxa::Format format)
        {
            switch (format)
            {
            case daxa::Format::R8G8B8A8_UINT: return 100;
            default: return daxa::default_format_score(format);
            }
        },
        .present_mode = daxa::PresentMode::DO_NOT_WAIT_FOR_VBLANK,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .debug_name = "Mandelbrot Swapchain",
    });

    daxa::PipelineCompiler pipeline_compiler = device.create_pipeline_compiler({
        .root_paths = {
            "shaders",
            ".out/debug/vcpkg_installed/x64-windows/include",
        },
        .debug_name = "Mandelbrot Pipeline Compiler",
    });

    daxa::ImGuiRenderer imgui_renderer = create_imgui_renderer();
    auto create_imgui_renderer() -> daxa::ImGuiRenderer
    {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(glfw_window_ptr, true);
        return daxa::ImGuiRenderer({
            .device = device,
            .pipeline_compiler = pipeline_compiler,
            .format = swapchain.get_format(),
        });
    }

    // clang-format off
    daxa::ComputePipeline compute_pipeline = pipeline_compiler.create_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"compute.hlsl"}},
        .push_constant_size = sizeof(ComputePush),
        .debug_name = "Mandelbrot Compute Pipeline",
    }).value();
    // clang-format on

    daxa::BufferId compute_input_buffer = device.create_buffer(daxa::BufferInfo{
        .size = sizeof(ComputeInput),
        .debug_name = "Mandelbrot Compute Input buffer",
    });
    ComputeInput compute_input = {};

    daxa::ImageId render_image = device.create_image(daxa::ImageInfo{
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {size_x, size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .debug_name = "Mandelbrot Render Image",
    });

    daxa::BinarySemaphore binary_semaphore = device.create_binary_semaphore({
        .debug_name = "Mandelbrot Present Semaphore",
    });

    static inline constexpr u64 FRAMES_IN_FLIGHT = 1;
    daxa::TimelineSemaphore gpu_framecount_timeline_sema = device.create_timeline_semaphore(daxa::TimelineSemaphoreInfo{
        .initial_value = 0,
        .debug_name = "Mandelbrot gpu framecount Timeline Semaphore",
    });
    u64 cpu_framecount = FRAMES_IN_FLIGHT - 1;

    Clock::time_point start = Clock::now();

    bool should_resize = false;

    App() : AppWindow<App>("Samples: Mandelbrot") {}

    ~App()
    {
        ImGui_ImplGlfw_Shutdown();
        device.destroy_buffer(compute_input_buffer);
        device.destroy_image(render_image);
    }

    bool update()
    {
        glfwPollEvents();
        if (glfwWindowShouldClose(glfw_window_ptr))
        {
            return true;
        }

        if (!minimized)
        {
            draw();
        }
        else
        {
            using namespace std::literals;
            std::this_thread::sleep_for(1ms);
        }

        return false;
    }

    void ui_update()
    {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Test");
        ImGui::DragFloat2("View Origin", reinterpret_cast<f32 *>(&compute_input.view_origin), 0.001f, -2.0f, 2.0f, "%.7f");
        ImGui::DragFloat("Zoom", &compute_input.zoom, 0.01f, 0.0f, 4.0f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::DragInt("Max Steps", &compute_input.max_steps, 1.0f, 1, 1024, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::End();
        ImGui::Render();
    }

    void draw()
    {
        ui_update();

        if (pipeline_compiler.check_if_sources_changed(compute_pipeline))
        {
            auto new_pipeline = pipeline_compiler.recreate_compute_pipeline(compute_pipeline);
            if (new_pipeline.is_ok())
            {
                compute_pipeline = new_pipeline.value();
            }
            else
            {
                std::cout << new_pipeline.message() << std::endl;
            }
        }

        if (should_resize)
        {
            do_resize();
        }

        auto swapchain_image = swapchain.acquire_next_image();

        auto cmd_list = device.create_command_list({
            .debug_name = "Mandelbrot Command List",
        });

        auto now = Clock::now();
        auto elapsed = std::chrono::duration<float>(now - start).count();

        auto compute_input_staging_buffer = device.create_buffer({
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .size = sizeof(ComputeInput),
            .debug_name = "Mandelbrot Compute Input Staging buffer " + std::to_string(cpu_framecount),
        });
        cmd_list.destroy_buffer_deferred(compute_input_staging_buffer);

        compute_input.time = elapsed;
        compute_input.frame_dim = u32vec2{{size_x, size_y}};
        auto buffer_ptr = device.map_memory_as<ComputeInput>(compute_input_staging_buffer);
        *buffer_ptr = compute_input;
        device.unmap_memory(compute_input_staging_buffer);

        cmd_list.pipeline_barrier({
            .awaited_pipeline_access = daxa::AccessConsts::HOST_WRITE,
            .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_READ,
        });

        cmd_list.copy_buffer_to_buffer({
            .src_buffer = compute_input_staging_buffer,
            .dst_buffer = compute_input_buffer,
            .size = sizeof(ComputeInput),
        });

        cmd_list.pipeline_barrier({
            .awaited_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
            .waiting_pipeline_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
        });

        cmd_list.set_pipeline(compute_pipeline);
        cmd_list.push_constant(ComputePush{
            .image_id = render_image.default_view(),
            .input_buffer_id = compute_input_buffer,
        });
        cmd_list.dispatch((size_x + 7) / 8, (size_y + 7) / 8);

        cmd_list.pipeline_barrier({
            .waiting_pipeline_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
        });

        cmd_list.pipeline_barrier_image_transition({
            .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
            .before_layout = daxa::ImageLayout::UNDEFINED,
            .after_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_id = swapchain_image,
        });

        cmd_list.pipeline_barrier_image_transition({
            .awaited_pipeline_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
            .waiting_pipeline_access = daxa::AccessConsts::TRANSFER_READ,
            .before_layout = daxa::ImageLayout::UNDEFINED,
            .after_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .image_id = render_image,
        });

        cmd_list.blit_image_to_image({
            .src_image = render_image,
            .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .dst_image = swapchain_image,
            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .src_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
            .src_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
            .dst_slice = {.image_aspect = daxa::ImageAspectFlagBits::COLOR},
            .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
        });

        imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, size_x, size_y);

        cmd_list.pipeline_barrier_image_transition({
            .awaited_pipeline_access = daxa::AccessConsts::TRANSFER_WRITE,
            .before_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .after_layout = daxa::ImageLayout::PRESENT_SRC,
            .image_id = swapchain_image,
        });

        cmd_list.pipeline_barrier_image_transition({
            .awaited_pipeline_access = daxa::AccessConsts::TRANSFER_READ,
            .waiting_pipeline_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
            .before_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .after_layout = daxa::ImageLayout::GENERAL,
            .image_id = render_image,
        });

        cmd_list.complete();

        ++cpu_framecount;
        device.submit_commands({
            .command_lists = {std::move(cmd_list)},
            .signal_binary_semaphores = {binary_semaphore},
            .signal_timeline_semaphores = {{gpu_framecount_timeline_sema, cpu_framecount}},
        });

        device.present_frame({
            .wait_binary_semaphores = {binary_semaphore},
            .swapchain = swapchain,
        });

        gpu_framecount_timeline_sema.wait_for_value(cpu_framecount - 1);
    }

    void on_mouse_move(f32, f32)
    {
    }

    void on_key(int, int)
    {
    }

    void on_resize(u32 sx, u32 sy)
    {
        size_x = sx;
        size_y = sy;
        minimized = (sx == 0 || sy == 0);

        if (!minimized)
        {
            should_resize = true;
            do_resize();
        }
    }

    void do_resize()
    {
        should_resize = false;
        device.destroy_image(render_image);
        render_image = device.create_image({
            .format = daxa::Format::R8G8B8A8_UNORM,
            .size = {size_x, size_y, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        });
        swapchain.resize(size_x, size_y);
        draw();
    }
};

int main()
{
    App app = {};
    while (true)
    {
        if (app.update())
            break;
    }
}
