#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    x.max(min).min(max)
}

#[inline]
fn apply_exposure_l(l: f32, ev: f32) -> f32 {
    let scale = ev.exp2(); // same as exp2(ev)
    clamp(l * scale, 0.0, 1.0)
}

pub fn exposure_inplace(pixels: &mut [f32], ev: f32) {
    assert!(
        pixels.len() % 4 == 0,
        "pixels must contain groups of 4 floats: L, a, b, alpha"
    );

    for p in pixels.chunks_exact_mut(4) {
        // p[0] = L
        // p[1] = a
        // p[2] = b
        // p[3] = alpha

        p[0] = apply_exposure_l(p[0], ev);
    }
}
