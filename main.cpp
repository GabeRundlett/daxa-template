#include "window.hpp"

#include <thread>
#include <iostream>
#include <cmath>

#define APPNAME "Daxa Template App"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

#define DAXA_GLSL 1
#define DAXA_HLSL 0

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>

using namespace daxa::types;
#include "shaders/shared.inl"

using Clock = std::chrono::high_resolution_clock;

struct App : AppWindow<App>
{
    daxa::Context daxa_ctx = daxa::create_context({
        .enable_validation = true,
    });
    daxa::Device device = daxa_ctx.create_device({
        .debug_name = APPNAME_PREFIX("device"),
    });

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
        .debug_name = APPNAME_PREFIX("swapchain"),
    });

    daxa::PipelineCompiler pipeline_compiler = device.create_pipeline_compiler({
        .shader_compile_options = {
            .root_paths = {
                DAXA_SHADER_INCLUDE_DIR,
                "shaders",
            },
#if DAXA_GLSL
            .language = daxa::ShaderLanguage::GLSL,
#elif DAXA_HLSL
            .language = daxa::ShaderLanguage::HLSL,
#endif
        },
        .debug_name = APPNAME_PREFIX("pipeline_compiler"),
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

    Clock::time_point start = Clock::now(), prev_time = start;
    f32 elapsed_s = 1.0f;

    // clang-format off
    daxa::ComputePipeline compute_pipeline = pipeline_compiler.create_compute_pipeline({
#if DAXA_GLSL
        .shader_info = {.source = daxa::ShaderFile{"compute.glsl"}},
#elif DAXA_HLSL
        .shader_info = {.source = daxa::ShaderFile{"compute.hlsl"}},
#endif
        .push_constant_size = sizeof(ComputePush),
        .debug_name = APPNAME_PREFIX("compute_pipeline"),
    }).value();
    // clang-format on

    GpuInput gpu_input = {
        .view_origin = {0, 0},
        .mouse_pos = {0, 0},
        .zoom = 2.0f,
        .max_steps = 512,
    };

    daxa::BufferId gpu_input_buffer = device.create_buffer({
        .size = sizeof(GpuInput),
        .debug_name = APPNAME_PREFIX("gpu_input_buffer"),
    });
    daxa::BufferId staging_gpu_input_buffer = device.create_buffer({
        .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .size = sizeof(GpuInput),
        .debug_name = APPNAME_PREFIX("staging_gpu_input_buffer"),
    });
    daxa::TaskBufferId task_gpu_input_buffer;
    daxa::TaskBufferId task_staging_gpu_input_buffer;

    daxa::CommandSubmitInfo submit_info;

    daxa::ImageId render_image = device.create_image(daxa::ImageInfo{
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {size_x, size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .debug_name = APPNAME_PREFIX("render_image"),
    });
    daxa::TaskImageId task_render_image;
    daxa::ImageId swapchain_image;
    daxa::TaskImageId task_swapchain_image;
    daxa::TaskList loop_task_list = record_loop_task_list();

    App() : AppWindow<App>(APPNAME) {}

    ~App()
    {
        device.wait_idle();
        device.collect_garbage();
        ImGui_ImplGlfw_Shutdown();
        device.destroy_buffer(gpu_input_buffer);
        device.destroy_buffer(staging_gpu_input_buffer);
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
            on_update();
        }
        else
        {
            using namespace std::literals;
            std::this_thread::sleep_for(1ms);
        }

        return false;
    }

    auto reload_pipeline(auto &pipeline) -> bool {
        if (pipeline_compiler.check_if_sources_changed(pipeline)) {
            auto new_pipeline = pipeline_compiler.recreate_compute_pipeline(pipeline);
            std::cout << new_pipeline.to_string() << std::endl;
            if (new_pipeline.is_ok()) {
                pipeline = new_pipeline.value();
                return true;
            }
        }
        return false;
    }

    void on_update()
    {
        auto now = Clock::now();
        elapsed_s = std::chrono::duration<f32>(now - prev_time).count();
        prev_time = now;

        gpu_input.time = elapsed_s;
        gpu_input.frame_dim = {size_x, size_y};

        ui_update();

        reload_pipeline(compute_pipeline);

        swapchain_image = swapchain.acquire_next_image();
        if (swapchain_image.is_empty())
            return;
        loop_task_list.execute();
    }

    void on_mouse_move(f32 x, f32 y)
    {
        gpu_input.mouse_pos = {x, y};
    }
    void on_mouse_scroll(f32 x, f32 y)
    {
        f32 mul = 0;
        if (y < 0)
            mul = pow(1.05f, abs(y));
        else if (y > 0)
            mul = 1.0f / pow(1.05f, abs(y));
        gpu_input.zoom *= mul;
    }
    void on_mouse_button(int, int)
    {
    }
    void on_key(int, int)
    {
    }
    void on_resize(u32 sx, u32 sy)
    {
        minimized = (sx == 0 || sy == 0);
        if (!minimized)
        {
            swapchain.resize();
            size_x = swapchain.info().width;
            size_y = swapchain.info().height;
            device.destroy_image(render_image);
            render_image = device.create_image({
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {size_x, size_y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_READ_WRITE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            });
            on_update();
        }
    }

    void ui_update()
    {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Test");
        ImGui::DragFloat2("View Origin", reinterpret_cast<f32 *>(&gpu_input.view_origin), 0.001f, -2.0f, 2.0f, "%.7f");
        ImGui::DragFloat("Zoom", &gpu_input.zoom, 0.01f, 0.0f, 4.0f, "%.7f", ImGuiSliderFlags_Logarithmic);
        ImGui::DragInt("Max Steps", &gpu_input.max_steps, 1.0f, 1, 1024, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::End();
        ImGui::Render();
    }

    auto record_loop_task_list() -> daxa::TaskList
    {
        daxa::TaskList new_task_list = daxa::TaskList({
            .device = device,
            .dont_use_split_barriers = true,
            .swapchain = swapchain,
            .debug_name = APPNAME_PREFIX("task_list"),
        });
        task_swapchain_image = new_task_list.create_task_image({
            .image = &swapchain_image,
            .swapchain_image = true,
            .debug_name = APPNAME_PREFIX("task_swapchain_image"),
        });
        task_render_image = new_task_list.create_task_image({
            .image = &render_image,
            .debug_name = APPNAME_PREFIX("task_render_image"),
        });

        task_gpu_input_buffer = new_task_list.create_task_buffer({
            .buffer = &gpu_input_buffer,
            .debug_name = APPNAME_PREFIX("task_gpu_input_buffer"),
        });
        task_staging_gpu_input_buffer = new_task_list.create_task_buffer({
            .buffer = &staging_gpu_input_buffer,
            .debug_name = APPNAME_PREFIX("task_staging_gpu_input_buffer"),
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::HOST_TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskRuntime /* runtime */)
            {
                GpuInput * buffer_ptr = device.map_memory_as<GpuInput>(staging_gpu_input_buffer);
                *buffer_ptr = this->gpu_input;
                device.unmap_memory(staging_gpu_input_buffer);
            },
            .debug_name = APPNAME_PREFIX("Input MemMap"),
        });
        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_WRITE},
                {task_staging_gpu_input_buffer, daxa::TaskBufferAccess::TRANSFER_READ},
            },
            .task = [this](daxa::TaskRuntime runtime)
            {
                auto cmd_list = runtime.get_command_list();
                cmd_list.copy_buffer_to_buffer({
                    .src_buffer = staging_gpu_input_buffer,
                    .dst_buffer = gpu_input_buffer,
                    .size = sizeof(GpuInput),
                });
            },
            .debug_name = APPNAME_PREFIX("Input Transfer"),
        });

        new_task_list.add_task({
            .used_buffers = {
                {task_gpu_input_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            },
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::COMPUTE_SHADER_WRITE_ONLY},
            },
            .task = [this](daxa::TaskRuntime runtime)
            {
                auto cmd_list = runtime.get_command_list();
                cmd_list.set_pipeline(compute_pipeline);
                cmd_list.push_constant(ComputePush {
                    .image_id = render_image.default_view(),
#if DAXA_GLSL
                    .gpu_input = this->device.buffer_reference(gpu_input_buffer),
#elif DAXA_HLSL
                    .input_buffer_id = gpu_input_buffer,
#endif
                });
                cmd_list.dispatch((size_x + 7) / 8, (size_y + 7) / 8);
            },
            .debug_name = APPNAME_PREFIX("Compute Task"),
        });

        new_task_list.add_task({
            .used_images = {
                {task_render_image, daxa::TaskImageAccess::TRANSFER_READ},
                {task_swapchain_image, daxa::TaskImageAccess::TRANSFER_WRITE},
            },
            .task = [this](daxa::TaskRuntime runtime)
            {
                auto cmd_list = runtime.get_command_list();
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
            },
            .debug_name = APPNAME_PREFIX("Blit Task (render to swapchain)"),
        });

        new_task_list.add_task({
            .used_images = {
                {task_swapchain_image, daxa::TaskImageAccess::COLOR_ATTACHMENT},
            },
            .task = [this](daxa::TaskRuntime runtime)
            {
                auto cmd_list = runtime.get_command_list();
                imgui_renderer.record_commands(ImGui::GetDrawData(), cmd_list, swapchain_image, size_x, size_y);
            },
            .debug_name = APPNAME_PREFIX("ImGui Task"),
        });

        new_task_list.submit(&submit_info);
        new_task_list.present({});
        new_task_list.complete();

        return new_task_list;
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
