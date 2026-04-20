use image::{Rgba, RgbaImage, ImageBuffer, DynamicImage};
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::types::cl_device_type;

use opencl_project::{ImageFilter::*, OpenClRuntime, ImageState};

const DEVICE_TYPE: cl_device_type = CL_DEVICE_TYPE_GPU;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = "test_images/img_firefox.png";
    let output_path = "test_images/out.png";

    let device_ids = get_all_devices(DEVICE_TYPE)?;
    let device_id = *device_ids
        .first()
        .ok_or("no OpenCL GPU device found")?;

    let device = Device::new(device_id);
    println!("Using device: {}", device.name()?);

    let runtime = OpenClRuntime::init(device_id)?;
    println!("OpenCL program compiled successfully.");
    println!("Kernel set initialized.");

    let img = image::open(image_path)?;
    let rgba8 = img.to_rgba8();
    let (width, height) = rgba8.dimensions();

    let rgba_pixels = rgba8.into_raw();

    let mut image_state: ImageState = ImageState::from_rgba_host(&runtime, width, height, &rgba_pixels)?;

    println!("Loaded image: {}x{}", width, height);
    println!(
        "Created ImageState for {} pixels.",
        width as usize * height as usize
    );

    //image_state.run_filter(HueShift { degrees: 60.0 }, &runtime)?;


    let out_rgba_pixels: Vec<u8> = image_state.to_rgba_host(&runtime)?;

    let out_rgba: RgbaImage =
        ImageBuffer::from_raw(width as u32, height as u32, out_rgba_pixels)
            .ok_or("failed to rebuild output image")?;

    out_rgba.save(output_path)?;

    println!("Saved filtered image to {}", output_path);

    Ok(())
}
