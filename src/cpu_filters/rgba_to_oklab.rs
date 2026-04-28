#[inline]
fn srgb_to_linear_1(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear_1(c[0]),
        srgb_to_linear_1(c[1]),
        srgb_to_linear_1(c[2]),
    ]
}

#[inline]
fn linear_srgb_to_oklab(c: [f32; 3]) -> [f32; 3] {
    let l = 0.412_221_46 * c[0] + 0.536_332_55 * c[1] + 0.051_445_995 * c[2];
    let m = 0.211_903_5 * c[0] + 0.680_699_5 * c[1] + 0.107_396_96 * c[2];
    let s = 0.088_302_46 * c[0] + 0.281_718_85 * c[1] + 0.629_978_7 * c[2];

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

pub fn rgba8_to_oklab(src: &[u8]) -> Vec<f32> {
    assert!(
        src.len() % 4 == 0,
        "src length must be divisible by 4 (RGBA8)"
    );

    let pixel_count = src.len() / 4;
    let mut dst = vec![0.0_f32; pixel_count * 4];

    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let srgba = [
            src_px[0] as f32 / 255.0,
            src_px[1] as f32 / 255.0,
            src_px[2] as f32 / 255.0,
            src_px[3] as f32 / 255.0,
        ];

        let linear = srgb_to_linear([srgba[0], srgba[1], srgba[2]]);
        let lab = linear_srgb_to_oklab(linear);

        dst_px[0] = lab[0];
        dst_px[1] = lab[1];
        dst_px[2] = lab[2];
        dst_px[3] = srgba[3];
    }

    dst
}
