inline float apply_contrast_l(float L, float amount)
{
    const float pivot = 0.5f;
    return clamp(pivot + (L - pivot) * amount, 0.0f, 1.0f);
}

__kernel void contrast(
    __global float4* pixels,
    float amount,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    float4 p = pixels[i];
    p.x = apply_contrast_l(p.x, amount); // p.x == L
    pixels[i] = p;
}
