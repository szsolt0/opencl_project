pub fn saturation_inplace(pixels: &mut [f32], amount: f32) {
    assert!(
        pixels.len() % 4 == 0,
        "pixels must contain groups of 4 floats: L, a, b, alpha"
    );

    for p in pixels.chunks_exact_mut(4) {
        // p[0] = L
        // p[1] = a
        // p[2] = b
        // p[3] = alpha

        p[1] *= amount; // a
        p[2] *= amount; // b
    }
}
