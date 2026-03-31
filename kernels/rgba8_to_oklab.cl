inline float srgb_to_linear_1(float x)
{
    return (x <= 0.04045f)
        ? (x / 12.92f)
        : pow((x + 0.055f) / 1.055f, 2.4f);
}

inline float3 srgb_to_linear(float3 c)
{
    return (float3)(
        srgb_to_linear_1(c.x),
        srgb_to_linear_1(c.y),
        srgb_to_linear_1(c.z)
    );
}

inline float3 linear_srgb_to_oklab(float3 c)
{
    float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
    float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
    float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;

    float l_ = cbrt(l);
    float m_ = cbrt(m);
    float s_ = cbrt(s);

    return (float3)(
        0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
        1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
        0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_
    );
}

__kernel void rgba8_to_oklab(
    __global const uchar4* src,
    __global float4* dst,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    uchar4 p = src[i];

    float4 srgba = convert_float4(p) / 255.0f;
    float3 linear = srgb_to_linear(srgba.xyz);
    float3 lab = linear_srgb_to_oklab(linear);

    dst[i] = (float4)(lab.x, lab.y, lab.z, srgba.w);
}
