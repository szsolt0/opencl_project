#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    x.max(min).min(max)
}

#[inline]
fn apply_contrast_l(l: f32, amount: f32) -> f32 {
    let pivot = 0.5_f32;
    clamp(pivot + (l - pivot) * amount, 0.0, 1.0)
}

pub fn contrast_inplace(pixels: &mut [f32], amount: f32) {
    assert!(
        pixels.len() % 4 == 0,
        "pixels must contain groups of 4 floats: L, a, b, alpha"
    );

    for p in pixels.chunks_exact_mut(4) {
        // p[0] = L
        p[0] = apply_contrast_l(p[0], amount);
    }
}
