use std::f32::consts::TAU;

#[inline]
fn wrap_angle_rad(mut h: f32) -> f32 {
    h = h % TAU;

    if h < 0.0 {
        h += TAU;
    }

    h
}

#[inline]
fn oklab_to_oklch(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let c = a.hypot(b);
    let h = wrap_angle_rad(b.atan2(a));

    (l, c, h)
}

#[inline]
fn oklch_to_oklab(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let a = c * h.cos();
    let b = c * h.sin();

    (l, a, b)
}

pub fn hue_shift_inplace(pixels: &mut [f32], hue_shift_rad: f32) {
    assert!(
        pixels.len() % 4 == 0,
        "pixels must contain groups of 4 floats: L, a, b, alpha"
    );

    for p in pixels.chunks_exact_mut(4) {
        let (l, c, mut h) = oklab_to_oklch(p[0], p[1], p[2]);

        if c > 1e-6 {
            h = wrap_angle_rad(h + hue_shift_rad);

            let (l, a, b) = oklch_to_oklab(l, c, h);

            p[0] = l;
            p[1] = a;
            p[2] = b;
            // p[3] preserved
        }
    }
}
