inline float wrap_angle_rad(float h)
{
    const float two_pi = 6.28318530717958647692f;

    h = fmod(h, two_pi);
    if (h < 0.0f) h += two_pi;

    return h;
}

inline float3 oklab_to_oklch(float3 lab)
{
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;

    float C = hypot(a, b);
    float h = atan2(b, a);   // [-pi, pi]
    h = wrap_angle_rad(h);   // [0, 2pi)

    return (float3)(L, C, h);
}

inline float3 oklch_to_oklab(float3 lch)
{
    float L = lch.x;
    float C = lch.y;
    float h = lch.z;

    float a = C * cos(h);
    float b = C * sin(h);

    return (float3)(L, a, b);
}

__kernel void hue_shift(
    __global float4* pixels,
    float hue_shift_rad,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    float4 p = pixels[i];
    float3 lab = p.xyz;

    float3 lch = oklab_to_oklch(lab);

    // Achromatic colors have C = 0, so hue is powerless there.
    // Skipping tiny chroma values also avoids angle noise near gray.
    if (lch.y > 1e-6f) {
        lch.z = wrap_angle_rad(lch.z + hue_shift_rad);
        lab = oklch_to_oklab(lch);
        pixels[i] = (float4)(lab.x, lab.y, lab.z, p.w);
    }
}
