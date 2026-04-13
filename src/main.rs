use image::GenericImageView;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::types::cl_device_type;

use opencl_project::image_filter::*;
use opencl_project::image_state::*;
use opencl_project::opencl_runtime::*;

const DEVICE_TYPE: cl_device_type = CL_DEVICE_TYPE_GPU;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = "test_images/img_land.jpg";

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

    let rgba_pixels: Vec<u32> = rgba8
        .pixels()
        .map(|p| {
            let [r, g, b, a] = p.0;
            (r as u32)
                | ((g as u32) << 8)
                | ((b as u32) << 16)
                | ((a as u32) << 24)
        })
        .collect();

    let image_state = ImageState::from_rgba_host(&runtime, width, height, &rgba_pixels)?;

    println!("Loaded image: {}x{}", width, height);
    println!(
        "Created ImageState for {} pixels.",
        width as usize * height as usize
    );

    let _ = image_state;

    Ok(())
}
