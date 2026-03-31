inline float linear_to_srgb_1(float x)
{
    x = clamp(x, 0.0f, 1.0f);

    return (x <= 0.0031308f)
        ? (12.92f * x)
        : (1.055f * pow(x, 1.0f / 2.4f) - 0.055f);
}

inline float3 linear_to_srgb(float3 c)
{
    return (float3)(
        linear_to_srgb_1(c.x),
        linear_to_srgb_1(c.y),
        linear_to_srgb_1(c.z)
    );
}

inline float3 oklab_to_linear_srgb(float3 lab)
{
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;

    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    return (float3)(
        +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
        -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
        -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s
    );
}

__kernel void oklab_to_rgba8(
    __global const float4* src,
    __global uchar4* dst,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    float4 p = src[i];

    float3 lab = p.xyz;
    float alpha = clamp(p.w, 0.0f, 1.0f);

    float3 linear_rgb = oklab_to_linear_srgb(lab);
    float3 srgb = linear_to_srgb(linear_rgb);

    uchar r = convert_uchar_sat_rte(srgb.x * 255.0f);
    uchar g = convert_uchar_sat_rte(srgb.y * 255.0f);
    uchar b = convert_uchar_sat_rte(srgb.z * 255.0f);
    uchar a = convert_uchar_sat_rte(alpha  * 255.0f);

    dst[i] = (uchar4)(r, g, b, a);
}
