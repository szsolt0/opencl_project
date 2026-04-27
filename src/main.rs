use std::time::Instant;
use std::env;
use std::sync::Arc;

use image::{ImageBuffer, RgbaImage};
use opencl3::device::{get_all_devices, Device};
use opencl3::types::cl_device_type;

#[allow(unused_imports)]
use opencl3::device::{CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};

use opencl_project::image_filter::ImageFilter::*;
use opencl_project::image_filter::ImageFilter;
use opencl_project::image_state::ImageState;
use opencl_project::opencl_runtime::OpenClRuntime;

//const DEVICE_TYPE: cl_device_type = CL_DEVICE_TYPE_CPU;

fn parse_filter(spec: &str) -> Result<ImageFilter, String> {
    let (name, args) = spec
        .split_once(':')
        .ok_or_else(|| format!("missing ':' in filter spec: {spec}"))?;

    let values: Vec<&str> = args
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    match name {
        "exposure" => {
            let ev = parse_one_f32(name, &values)?;
            Ok(Exposure { ev })
        }

        "contrast" => {
            let amount = parse_one_f32(name, &values)?;
            Ok(Contrast { amount })
        }

        "saturation" => {
            let amount = parse_one_f32(name, &values)?;
            Ok(Saturation { amount })
        }

        "hue_shift" | "hueshift" | "hue" => {
            let degrees = parse_one_f32(name, &values)?;
            Ok(HueShift { degrees })
        }

        "gaussian_blur" | "gaussian" => {
            if values.len() != 2 {
                return Err(format!(
                    "{name} expects 2 arguments: radius,sigma"
                ));
            }

            let radius = values[0]
                .parse::<u32>()
                .map_err(|_| format!("invalid radius for {name}: {}", values[0]))?;

            let sigma = values[1]
                .parse::<f32>()
                .map_err(|_| format!("invalid sigma for {name}: {}", values[1]))?;

            Ok(GaussianBlur { radius, sigma })
        }

        "box_blur" | "box" => {
            if values.len() != 1 {
                return Err(format!("{name} expects 1 argument: radius"));
            }

            let radius = values[0]
                .parse::<u32>()
                .map_err(|_| format!("invalid radius for {name}: {}", values[0]))?;

            Ok(BoxBlur { radius })
        }

        _ => Err(format!("unknown filter: {name}")),
    }
}

fn parse_one_f32(name: &str, values: &[&str]) -> Result<f32, String> {
    if values.len() != 1 {
        return Err(format!("{name} expects 1 argument"));
    }

    values[0]
        .parse::<f32>()
        .map_err(|_| format!("invalid number for {name}: {}", values[0]))
}

fn print_usage(program_name: &str) {
    eprintln!(
        "Usage:
  {program_name} <device> <input> <output> <filter>...

Filters:
  exposure:<ev>
  contrast:<amount>
  saturation:<amount>
  hue_shift:<degrees>
  gaussian_blur:<radius>,<sigma>
  box_blur:<radius>

Example:
  {program_name} gpu input.png output.png hue_shift:45 saturation:1.5 gaussian_blur:4,2.5
Device types:
  gpu, cpu"
    );
}

fn parse_device_type(name: &str) -> Result<cl_device_type, String> {
    match name {
        "cpu" => Ok(CL_DEVICE_TYPE_CPU),
        "gpu" => Ok(CL_DEVICE_TYPE_GPU),
        _ => Err(format!("unknown filter: {name}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        print_usage(&args[0]);
        return Err("not enough arguments".into());
    }

    let device_type = parse_device_type(&args[1])?;
    let image_path = &args[2];
    let output_path = &args[3];

    let filters: Vec<ImageFilter> = args[4..]
        .iter()
        .map(|arg| parse_filter(arg))
        .collect::<Result<_, _>>()?;

    let device_ids = get_all_devices(device_type)?;
    let device_id = *device_ids
        .first()
        .ok_or("no OpenCL device found")?;

    let device = Device::new(device_id);
    println!("Using device: {}", device.name()?);

    let runtime_start = Instant::now();
    let runtime = OpenClRuntime::init(device_id)?;
    let runtime_load_time = runtime_start.elapsed();
    println!(
        "bench: OpenCL runtime/kernel load: {:.3} ms",
        runtime_load_time.as_secs_f64() * 1000.0
    );

    let runtime = Arc::new(runtime);

    let img = image::open(image_path)?;
    let rgba8 = img.to_rgba8();
    let (width, height) = rgba8.dimensions();

    let mut rgba_pixels = rgba8.into_raw();

    println!("Loaded image: {}x{}", width, height);
    println!(
        "Created ImageState for {} pixels.",
        width as usize * height as usize
    );

    // The actual total work, not counting the PNG/JPG <-> Raw conversion
    // and disk I/O.
    let total_image_work = Instant::now();

    let mut image_state =
        ImageState::from_rgba_host(runtime.clone(), width, height, &rgba_pixels)?;

    for filter in filters {
        // this is already written by benchmark code
        //println!("Running filter: {:?}", filter);
        image_state.run_filter(filter)?;
    }

    image_state.to_rgba_host(&mut rgba_pixels)?;

    println!("bench: FULL TIME: {:.3} ms", total_image_work.elapsed().as_secs_f64() * 1000.0);

    let out_rgba: RgbaImage =
        ImageBuffer::from_raw(width, height, rgba_pixels)
            .ok_or("failed to rebuild output image")?;

    out_rgba.save(output_path)?;

    println!("Saved filtered image to {}", output_path);

    Ok(())
}
