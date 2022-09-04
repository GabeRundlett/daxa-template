#pragma once

#include <daxa/daxa.inl>

struct GpuInput
{
    u32vec2 frame_dim;
    f32vec2 view_origin;
    f32vec2 mouse_pos;
    f32 zoom;
    f32 time;
    i32 max_steps;
};
DAXA_REGISTER_STRUCT_GET_BUFFER(GpuInput);

struct ComputePush
{
    ImageViewId image_id;
    BufferId input_buffer_id;
};
