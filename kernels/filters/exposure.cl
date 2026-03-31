inline float apply_exposure_l(float L, float ev)
{
    float scale = exp2(ev);
    return clamp(L * scale, 0.0f, 1.0f);
}

__kernel void exposure(
    __global float4* pixels,
    float ev,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    float4 p = pixels[i];

    // p.x = L
    // p.y = a
    // p.z = b
    // p.w = alpha

    p.x = apply_exposure_l(p.x, ev);

    pixels[i] = p;
}
