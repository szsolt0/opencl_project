use opencl3::memory::Buffer;
use opencl3::command_queue::CommandQueue;
use opencl3::kernel::Kernel;
use opencl3::program::Program;

pub struct ImageView<'a, T> {
    pub data: &'a [T],
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

pub struct ImageViewMut<'a, T> {
    pub data: &'a mut [T],
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TileRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

pub enum ImageFilter {
    Exposure { ev: f32 },
    Contrast { amount: f32 },
    Saturation { amount: f32 },
    HueShift { degrees: f32 },
    GaussianBlur { radius: u32, sigma: f32 },
    BoxBlur { radius: u32 },
}

impl ImageFilter {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Exposure { .. } => "exposure",
            Self::Contrast { .. } => "contrast",
            Self::Saturation { .. } => "saturation",
            Self::HueShift { .. } => "hue_shift",
            Self::GaussianBlur { .. } => "gaussian_blur",
            Self::BoxBlur { .. } => "box_blur",
        }
    }

    pub fn halo_size(&self) -> usize {
        match self {
            Self::Exposure { .. } => 0,
            Self::Contrast { .. } => 0,
            Self::Saturation { .. } => 0,
            Self::HueShift { .. } => 0,
            Self::GaussianBlur { radius, .. } => *radius as usize,
            Self::BoxBlur { radius } => *radius as usize,
        }
    }

    pub fn is_point_op(&self) -> bool {
        self.halo_size() == 0
    }
}

pub struct Rgba8ToOklabKernel {
    kernel: opencl3::kernel::Kernel,
}

pub struct OklabToRgba8Kernel {
    kernel: opencl3::kernel::Kernel,
}

pub struct HueShiftOklabKernel {
    kernel: opencl3::kernel::Kernel,
}

pub struct ContrastOklabKernel {
    kernel: opencl3::kernel::Kernel,
}

impl Rgba8ToOklabKernel {
    pub fn new(program: &Program) -> Result<Self, opencl3::error_codes::ClError> {
        let kernel = Kernel::create(program, "rgba8_to_oklab")?;
        Ok(Self { kernel })
    }

    pub fn run(
        &mut self,
        queue: &CommandQueue,
        src: &Buffer<u8>,
        dst: &Buffer<f32>,
        pixel_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixel_count_i32 = i32::try_from(pixel_count)?;

        unsafe {
            self.kernel.set_arg(0, src)?;
            self.kernel.set_arg(1, dst)?;
            self.kernel.set_arg(2, &pixel_count_i32)?;
        }

        let global_work_size = [pixel_count];
        unsafe {
            queue.enqueue_nd_range_kernel(
                self.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                std::ptr::null(),
                &[][..],
            )?;
        }

        Ok(())
    }
}

impl OklabToRgba8Kernel {
    pub fn new(program: &Program) -> Result<Self, opencl3::error_codes::ClError> {
        let kernel = Kernel::create(program, "oklab_to_rgba8")?;
        Ok(Self { kernel })
    }

    pub fn run(
        &mut self,
        queue: &CommandQueue,
        src: &Buffer<f32>,
        dst: &Buffer<u8>,
        pixel_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixel_count_i32 = i32::try_from(pixel_count)?;

        unsafe {
            self.kernel.set_arg(0, src)?;
            self.kernel.set_arg(1, dst)?;
            self.kernel.set_arg(2, &pixel_count_i32)?;
        }

        let global_work_size = [pixel_count];
        unsafe {
            queue.enqueue_nd_range_kernel(
                self.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                std::ptr::null(),
                &[][..],
            )?;
        }

        Ok(())
    }
}

impl HueShiftOklabKernel {
    pub fn new(program: &Program) -> Result<Self, opencl3::error_codes::ClError> {
        let kernel = Kernel::create(program, "hue_shift")?;
        Ok(Self { kernel })
    }

    pub fn run(
        &mut self,
        queue: &CommandQueue,
        pixels: &Buffer<f32>,
        amount: f32,
        pixel_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixel_count_i32 = i32::try_from(pixel_count)?;

        unsafe {
            self.kernel.set_arg(0, pixels)?;
            self.kernel.set_arg(1, &amount)?;
            self.kernel.set_arg(2, &pixel_count_i32)?;
        }

        let global_work_size = [pixel_count];
        unsafe {
            queue.enqueue_nd_range_kernel(
                self.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                std::ptr::null(),
                &[][..],
            )?;
        }

        Ok(())
    }
}

impl ContrastOklabKernel {
    pub fn new(program: &Program) -> Result<Self, opencl3::error_codes::ClError> {
        let kernel = Kernel::create(program, "contrast")?;
        Ok(Self { kernel })
    }

    pub fn run(
        &mut self,
        queue: &CommandQueue,
        pixels: &Buffer<f32>,
        amount: f32,
        pixel_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pixel_count_i32 = i32::try_from(pixel_count)?;

        unsafe {
            self.kernel.set_arg(0, pixels)?;
            self.kernel.set_arg(1, &amount)?;
            self.kernel.set_arg(2, &pixel_count_i32)?;
        }

        let global_work_size = [pixel_count];
        unsafe {
            queue.enqueue_nd_range_kernel(
                self.kernel.get(),
                1,
                std::ptr::null(),
                global_work_size.as_ptr(),
                std::ptr::null(),
                &[][..],
            )?;
        }

        Ok(())
    }
}


pub struct ColorKernels {
    pub rgba8_to_oklab: Rgba8ToOklabKernel,
    pub oklab_to_rgba8: OklabToRgba8Kernel,
    pub hue_shift_oklab: HueShiftOklabKernel,
    pub contrast_oklab: ContrastOklabKernel,
}

impl ColorKernels {
    pub fn new(program: &Program) -> Result<Self, opencl3::error_codes::ClError> {
        Ok(Self {
            rgba8_to_oklab: Rgba8ToOklabKernel::new(program)?,
            oklab_to_rgba8: OklabToRgba8Kernel::new(program)?,
            hue_shift_oklab: HueShiftOklabKernel::new(program)?,
            contrast_oklab: ContrastOklabKernel::new(program)?,
        })
    }
}
