use image::{DynamicImage, ImageBuffer, RgbaImage};
use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use std::error::Error;
use std::ptr;

use opencl_project::ColorKernels;

fn main() -> Result<(), Box<dyn Error>> {
    let input_path = "test_images/img_land.jpg";
    let output_path = "test_images/out.jpg";

    let shift_degrees = 0.0f32;
    let contrast = 1.0f32;
    let saturation = 1.0f32;
    let exposure = 1.0f32;

    let img = image::open(input_path)?.to_rgba8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let pixel_count = width * height;

    let input_pixels = img.into_raw();
    let mut output_pixels = vec![0u8; input_pixels.len()];

    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
    let device_id = *device_ids.first().ok_or("no OpenCL GPU device found")?;
    let device = Device::new(device_id);

    let context = Context::from_device(&device)?;
    let queue = CommandQueue::create_default(&context, 0)?;

    let mut kernels = ColorKernels::create(&context, None)?;

    let mut src_rgba = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_READ_ONLY,
            pixel_count * 4,
            ptr::null_mut(),
        )?
    };

    let tmp_oklab = unsafe {
        Buffer::<f32>::create(
            &context,
            CL_MEM_READ_WRITE,
            pixel_count * 4,
            ptr::null_mut(),
        )?
    };

    let dst_rgba = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_READ_WRITE,
            pixel_count * 4,
            ptr::null_mut(),
        )?
    };

    unsafe {
        queue.enqueue_write_buffer(&mut src_rgba, CL_BLOCKING, 0, &input_pixels, &[])?;
    }

    kernels
        .rgba8_to_oklab
        .run(&queue, &src_rgba, &tmp_oklab, pixel_count)?;

    kernels
        .hue_shift_oklab
        .run(&queue, &tmp_oklab, shift_degrees, pixel_count)?;

    kernels
        .contrast_oklab
        .run(&queue, &tmp_oklab, contrast, pixel_count)?;

    kernels
        .saturation_oklab
        .run(&queue, &tmp_oklab, saturation, pixel_count)?;

    kernels
        .exposure_oklab
        .run(&queue, &tmp_oklab, exposure, pixel_count)?;

    kernels
        .oklab_to_rgba8
        .run(&queue, &tmp_oklab, &dst_rgba, pixel_count)?;

    queue.finish()?;

    unsafe {
        queue.enqueue_read_buffer(&dst_rgba, CL_BLOCKING, 0, &mut output_pixels, &[])?;
    }

    let out_rgba: RgbaImage =
        ImageBuffer::from_raw(width as u32, height as u32, output_pixels)
            .ok_or("failed to rebuild output image")?;

    let out_rgb = DynamicImage::ImageRgba8(out_rgba).to_rgb8();
    out_rgb.save(output_path)?;

    println!("Wrote {}", output_path);
    Ok(())
}
