use std::time::Instant;
use std::env;
use std::sync::Arc;

use image::{ImageBuffer, RgbaImage};
use opencl3::device::{get_all_devices, Device};

#[allow(unused_imports)]
use opencl3::device::{CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};

use opencl_project::image_filter::ImageFilter::*;
use opencl_project::image_filter::ImageFilter;
use opencl_project::image_state::ImageState;
use opencl_project::opencl_runtime::OpenClRuntime;

use opencl_project::cpu_filters::point_ops::contrast::*;
use opencl_project::cpu_filters::point_ops::exposure::*;
use opencl_project::cpu_filters::point_ops::saturation::*;
use opencl_project::cpu_filters::oklab_to_rgba::*;
use opencl_project::cpu_filters::rgba_to_oklab::*;
use opencl_project::cpu_filters::point_ops::hue_shift::*;

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
  {program_name} <cpu|gpu> <input> <output> <filter>...

Filters:
  exposure:<ev>
  contrast:<amount>
  saturation:<amount>
  hue_shift:<degrees>
  gaussian_blur:<radius>,<sigma>
  box_blur:<radius>

Example:
  {program_name} gpu input.png output.png hue_shift:45 saturation:1.5 gaussian_blur:4,2.5"
    );
}

pub enum DeviceType {
    Gpu,
    Cpu,
}

fn parse_device_type(name: &str) -> Result<DeviceType, String> {
    match name {
        "cpu" => Ok(DeviceType::Cpu),
        "gpu" => Ok(DeviceType::Gpu),
        _ => Err(format!("unknown device: {name}")),
    }
}

struct CliArgs {
    device_type: DeviceType,
    input_path: String,
    output_path: String,
    filters: Vec<ImageFilter>,
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        print_usage(&args[0]);
        return Err("not enough arguments".into());
    }

    let device_type = parse_device_type(&args[1])?;
    let input_path = args[2].clone();
    let output_path = args[3].clone();

    let filters: Vec<ImageFilter> = args[4..]
        .iter()
        .map(|arg| parse_filter(arg))
        .collect::<Result<_, _>>()?;

    Ok(CliArgs {
        device_type,
        input_path,
        output_path,
        filters,
    })
}

fn gpu_main(
    width: u32,
    height: u32,
    filters: &[ImageFilter],
    rgba_pixels: &mut [u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
    let device_id = *device_ids
        .first()
        .ok_or("no OpenCL GPU device found")?;

    let device = Device::new(device_id);
    println!("Using GPU device: {}", device.name()?);

    let runtime_start = Instant::now();

    let runtime = OpenClRuntime::init(device_id)?;

    println!(
        "bench: OpenCL runtime/kernel load: {:.3} ms",
        runtime_start.elapsed().as_secs_f64() * 1000.0
    );

    let runtime = Arc::new(runtime);

    let mut image_state =
        ImageState::from_rgba_host(runtime, width, height, rgba_pixels)?;

    for &filter in filters {
        image_state.run_filter(filter)?;
    }

    image_state.to_rgba_host(rgba_pixels)?;

    Ok(())
}

fn cpu_main(
    width: u32,
    height: u32,
    filters: &[ImageFilter],
    rgba_pixels: &mut [u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let _pixel_count = width as usize * height as usize;

    let t = Instant::now();
    let mut oklab_buf = rgba8_to_oklab(rgba_pixels);
    println!("bench: cpu rgba -> oklab: {:.5} ms", t.elapsed().as_secs_f64() * 1000.0);

    for filter in filters {
        let t = Instant::now();

        match *filter {
            HueShift { degrees } => hue_shift_inplace(&mut oklab_buf, degrees.to_radians()),
            Exposure { ev } => exposure_inplace(&mut oklab_buf, ev),
            Contrast { amount } => contrast_inplace(&mut oklab_buf, amount),
            Saturation { amount } => saturation_inplace(&mut oklab_buf, amount),
            _ => panic!("AAAAAHHHHHHH!")
        }

        println!("bench: {:?}: {:.5} ms", filter, t.elapsed().as_secs_f64() * 1000.0);
    }

    let t = Instant::now();
    oklab_to_rgba8(&oklab_buf, rgba_pixels);
    println!("bench: cpu oklab -> rgba: {:.5} ms", t.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_args()?;

    let img = image::open(&cli.input_path)?;
    let rgba8 = img.to_rgba8();
    let (width, height) = rgba8.dimensions();

    let mut rgba_pixels = rgba8.into_raw();

    println!("Loaded image: {}x{}", width, height);
    println!(
        "Image has {} pixels.",
        width as usize * height as usize
    );

    let total_image_work = Instant::now();

    match cli.device_type {
        DeviceType::Gpu => {
            gpu_main(width, height, &cli.filters, &mut rgba_pixels)?;
        }
        DeviceType::Cpu => {
            cpu_main(width, height, &cli.filters, &mut rgba_pixels)?;
        }
    }

    println!(
        "bench: FULL TIME: {:.3} ms",
        total_image_work.elapsed().as_secs_f64() * 1000.0
    );

    let out_rgba: RgbaImage =
        ImageBuffer::from_raw(width, height, rgba_pixels)
            .ok_or("failed to rebuild output image")?;

    out_rgba.save(&cli.output_path)?;

    println!("Saved filtered image to {}", cli.output_path);

    Ok(())
}
