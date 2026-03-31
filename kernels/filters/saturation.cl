inline float2 scale_ab(float2 ab, float amount)
{
    return ab * amount;
}

__kernel void saturation_oklab(
    __global float4* pixels,
    float amount,
    int pixel_count)
{
    int i = get_global_id(0);
    if (i >= pixel_count) return;

    float4 p = pixels[i];

    // p.x = L
    // p.y = a
    // p.z = b
    // p.w = alpha

    float2 ab = (float2)(p.y, p.z);
    ab = scale_ab(ab, amount);

    p.y = ab.x;
    p.z = ab.y;

    pixels[i] = p;
}
