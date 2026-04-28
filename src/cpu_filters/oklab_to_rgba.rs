#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    x.max(min).min(max)
}

#[inline]
fn linear_to_srgb_1(mut x: f32) -> f32 {
    x = clamp(x, 0.0, 1.0);

    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn linear_to_srgb(c: [f32; 3]) -> [f32; 3] {
    [
        linear_to_srgb_1(c[0]),
        linear_to_srgb_1(c[1]),
        linear_to_srgb_1(c[2]),
    ]
}

#[inline]
fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let l = lab[0];
    let a = lab[1];
    let b = lab[2];

    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_35 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

#[inline]
fn f32_to_u8_sat_rte(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

pub fn oklab_to_rgba8(src: &[f32], dst: &mut [u8]) {
    assert!(
        src.len() % 4 == 0,
        "src length must be divisible by 4"
    );
    assert_eq!(
        src.len(),
        dst.len(),
        "src and dst must have the same element count"
    );

    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let lab = [src_px[0], src_px[1], src_px[2]];
        let alpha = clamp(src_px[3], 0.0, 1.0);

        let linear_rgb = oklab_to_linear_srgb(lab);
        let srgb = linear_to_srgb(linear_rgb);

        dst_px[0] = f32_to_u8_sat_rte(srgb[0] * 255.0);
        dst_px[1] = f32_to_u8_sat_rte(srgb[1] * 255.0);
        dst_px[2] = f32_to_u8_sat_rte(srgb[2] * 255.0);
        dst_px[3] = f32_to_u8_sat_rte(alpha * 255.0);
    }
}
