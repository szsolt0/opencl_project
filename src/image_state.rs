use opencl3::{
    command_queue::CL_BLOCKING, kernel::ExecuteKernel, memory::{Buffer, CL_MEM_READ_WRITE, ClMem}
};

use std::ptr;
use std::sync::Arc;

use crate::{ab_buffers::AbBuffers, image_filter::ImageFilter};
use crate::opencl_runtime::OpenClRuntime;

pub struct ImageState {
    pub width: u32,
    pub height: u32,
    pub buffers: AbBuffers<f32>,
    pub runtime: Arc<OpenClRuntime>,
}


impl ImageState {
    pub fn from_rgba_host(
        runtime: Arc<OpenClRuntime>,
        width: u32,
        height: u32,
        rgba_host: &[u8],
    ) -> Result<Self, ImageStateInitError> {
        let pixel_count = width as usize * height as usize;

        if rgba_host.len() != pixel_count*4 {
            return Err(ImageStateInitError::SizeMismatch {
                expected_pixels: pixel_count,
                actual_pixels: rgba_host.len(),
            });
        }

        let mut original_rgba = unsafe {
            Buffer::<u8>::create(
                &runtime.context,
                CL_MEM_READ_WRITE,
                pixel_count * 4,
                ptr::null_mut(),
            )?
        };

        let oklab_buffers: AbBuffers<f32> = AbBuffers::<f32>::create(4*pixel_count, &runtime.context)?;

        unsafe {
            runtime.queue.enqueue_write_buffer(
                &mut original_rgba,
                CL_BLOCKING,
                0,
                rgba_host,
                &[][..],
            )?;
        }

        unsafe {
            ExecuteKernel::new(&runtime.filter_kernels.rgba_to_oklab)
            .set_arg(&original_rgba.get())
            .set_arg(&oklab_buffers.current().get())
            .set_arg(&(pixel_count as i32))
            .set_global_work_size(pixel_count)
            .enqueue_nd_range(&runtime.queue)?;
        }

        runtime.queue.finish()?;
        std::mem::drop(original_rgba);

        Ok(Self {
            width,
            height,
            buffers: oklab_buffers,
            runtime,
        })
    }

    pub fn to_rgba_host(
        &mut self,
        rgba_host: &mut [u8],
    ) -> Result<(), ImageStateInitError> {
        let pixel_count = self.width as usize * self.height as usize;

        if rgba_host.len() != pixel_count*4 {
            return Err(ImageStateInitError::SizeMismatch {
                expected_pixels: pixel_count,
                actual_pixels: rgba_host.len(),
            });
        }

        let rgba_buffer = unsafe {
            Buffer::<u8>::create(
                &self.runtime.context,
                CL_MEM_READ_WRITE,
                pixel_count * 4,
                ptr::null_mut(),
            )?
        };

        println!("convert oklab to rgba");

        unsafe {
            ExecuteKernel::new(&self.runtime.filter_kernels.oklab_to_rgba)
            .set_arg(&self.buffers.current().get())
            .set_arg(&rgba_buffer.get())
            .set_arg(&(pixel_count as i32))
            .set_global_work_size(pixel_count)
            .enqueue_nd_range(&self.runtime.queue)?;
        }

        self.runtime.queue.finish()?;
        println!("convert oklab to rgba finished");

        unsafe {
            self.runtime.queue.enqueue_read_buffer(
                &rgba_buffer,
                CL_BLOCKING,
                0,
                rgba_host,
                &[][..],
            )?;
        }

        Ok(())
    }

    pub fn run_filter(
        &mut self,
        filter: ImageFilter,
    ) -> Result<(), ImageStateInitError> {
        let pixel_count = (self.width as usize) * (self.height as usize);
        let kernel = self.runtime.filter_kernels.kernel_for_filter(&filter);
        let mut exec_kernel = ExecuteKernel::new(&kernel);

        println!("running filter: {}", filter.kind().kernel_name());

        match filter {
            // Point ops: in-place on current buffer
            ImageFilter::Exposure { ev } => unsafe {
                exec_kernel
                .set_arg(&self.buffers.current().get())
                .set_arg(&ev)
                .set_arg(&(pixel_count as i32));
            },

            ImageFilter::Contrast { amount } => unsafe {
                exec_kernel
                .set_arg(&self.buffers.current().get())
                .set_arg(&amount)
                .set_arg(&(pixel_count as i32));
            },

            ImageFilter::Saturation { amount } => unsafe {
                exec_kernel
                .set_arg(&self.buffers.current().get())
                .set_arg(&amount)
                .set_arg(&(pixel_count as i32));
            },

            ImageFilter::HueShift { degrees } => unsafe {
                exec_kernel
                .set_arg(&self.buffers.current().get())
                .set_arg(&degrees.to_radians())
                .set_arg(&(pixel_count as i32));
            },

            // Spatial ops: current -> other, then flip
            /*ImageFilter::GaussianBlur { radius, sigma } => unsafe {
                exec_kernel
                .set_arg(&self.ab_buffers.current())
                .set_arg(&self.ab_buffers.other())
                .set_arg(&(self.width as i32))
                .set_arg(&(self.height as i32))
                .set_arg(&(radius as i32))
                .set_arg(&sigma);

                self.ab_buffers.swap();
            },

            ImageFilter::BoxBlur { radius } => unsafe {
                exec_kernel
                .set_arg(&self.ab_buffers.current())
                .set_arg(&self.ab_buffers.other())
                .set_arg(&(self.width as i32))
                .set_arg(&(self.height as i32))
                .set_arg(&(radius as i32));

                self.ab_buffers.swap();
            }*/

            _ => panic!("unimplemented")
        }

        unsafe {
            exec_kernel
            .set_global_work_size(pixel_count)
            .enqueue_nd_range(&self.runtime.queue)?;
        }

        //runtime.queue.finish()?;
        println!("conversion finished");
        Ok(())
    }
}

#[derive(Debug)]
pub enum ImageStateInitError {
    SizeMismatch {
        expected_pixels: usize,
        actual_pixels: usize,
    },
    OpenCl(opencl3::error_codes::ClError),
}

impl From<opencl3::error_codes::ClError> for ImageStateInitError {
    fn from(value: opencl3::error_codes::ClError) -> Self {
        Self::OpenCl(value)
    }
}

impl std::fmt::Display for ImageStateInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeMismatch {
                expected_pixels,
                actual_pixels,
            } => write!(
                f,
                "RGBA host buffer size mismatch: expected {} pixels, got {}",
                expected_pixels,
                actual_pixels
            ),
            Self::OpenCl(err) => write!(f, "OpenCL error: {:?}", err),
        }
    }
}

impl std::error::Error for ImageStateInitError {}
