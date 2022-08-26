#include "shared.inl"

[[vk::push_constant]] const ComputePush p;

#define SUBSAMPLES 2

float3 hsv2rgb(float3 c)
{
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * lerp(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

float3 mandelbrot_colored(float2 pixel_p)
{
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);
    float2 uv = pixel_p / float2(input[0].frame_dim.xy);
    uv = (uv - 0.5) * float2(float(input[0].frame_dim.x) / float(input[0].frame_dim.y), 1);
    float time = input[0].time;
    float scale = input[0].zoom;
    float2 center = input[0].view_origin;
    uint max_steps = input[0].max_steps;
    float2 z = uv * scale * 2 + center;
    float2 c = z;
    uint i = 0;
    for (; i < max_steps; ++i)
    {
        float2 z_ = z;
        z.x = z_.x * z_.x - z_.y * z_.y;
        z.y = 2.0 * z_.x * z_.y;
        z += c;
        if (dot(z, z) > 256 * 256)
            break;
    }
    float3 col = 0;
    if (i != max_steps)
    {
        float l = i;
        float sl = l - log2(log2(dot(z, z))) + 4.0;
        sl = pow(sl * 0.01, 1.0);
        col = hsv2rgb(float3(sl, 1, 1));
    }
    return col;
}

// clang-format off
[numthreads(8, 8, 1)] void main(uint3 pixel_i : SV_DispatchThreadID)
// clang-format on
{
    StructuredBuffer<GpuInput> input = daxa::get_StructuredBuffer<GpuInput>(p.input_buffer_id);
    RWTexture2D<float4> render_image = daxa::get_RWTexture2D<float4>(p.image_id);
    if (pixel_i.x >= input[0].frame_dim.x || pixel_i.y >= input[0].frame_dim.y)
        return;
    float3 col = 0;
    for (int yi = 0; yi < SUBSAMPLES; ++yi)
    {
        for (int xi = 0; xi < SUBSAMPLES; ++xi)
        {
            float2 offset = float2(xi, yi) / float(SUBSAMPLES);
            col += mandelbrot_colored(float2(pixel_i.xy) + offset);
        }
    }
    col *= 1.0 / float(SUBSAMPLES * SUBSAMPLES);
    render_image[pixel_i.xy] = float4(col, 1.0);
}
