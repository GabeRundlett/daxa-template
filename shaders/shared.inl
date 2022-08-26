#pragma once

#ifdef __cplusplus
#include <daxa/daxa.hpp>
#else
#include "daxa/daxa.hlsl"
#endif

struct GpuInput
{
    daxa::u32vec2 frame_dim;
    daxa::f32vec2 view_origin;
    daxa::f32vec2 mouse_pos;
    daxa::f32 zoom;
    daxa::f32 time;
    daxa::i32 max_steps;
};
DAXA_DEFINE_GET_STRUCTURED_BUFFER(GpuInput);

struct ComputePush
{
    daxa::ImageViewId image_id;
    daxa::BufferId input_buffer_id;
};
