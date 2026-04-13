use opencl3::memory::Buffer;

use crate::ab_buffers::AbBuffers;
use crate::opencl_runtime::OpenClRuntime;

pub struct ImageState {
    pub width: u32,
    pub height: u32,
    pub ab_buffers: AbBuffers<f32>,
    pub original_rgba: opencl3::memory::Buffer<u32>,
}

use std::ptr;

use opencl3::{
    command_queue::CL_BLOCKING,
    kernel::ExecuteKernel,
    memory::{CL_MEM_READ_WRITE},
};

impl ImageState {
    pub fn from_rgba_host(
        runtime: &OpenClRuntime,
        width: u32,
        height: u32,
        rgba_host: &[u32],
    ) -> Result<Self, ImageStateInitError> {
        let pixel_count = width as usize * height as usize;

        if rgba_host.len() != pixel_count {
            return Err(ImageStateInitError::SizeMismatch {
                expected_pixels: pixel_count,
                actual_pixels: rgba_host.len(),
            });
        }

        let mut original_rgba = unsafe {
            Buffer::<u32>::create(
                &runtime.context,
                CL_MEM_READ_WRITE,
                pixel_count,
                ptr::null_mut(),
            )?
        };

        let oklab_a = unsafe {
            Buffer::<f32>::create(
                &runtime.context,
                CL_MEM_READ_WRITE,
                pixel_count * 3,
                ptr::null_mut(),
            )?
        };

        let oklab_b = unsafe {
            Buffer::<f32>::create(
                &runtime.context,
                CL_MEM_READ_WRITE,
                pixel_count * 3,
                ptr::null_mut(),
            )?
        };

        let oklab = AbBuffers {
            a: oklab_a,
            b: oklab_b,
            a_is_current: true,
        };

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
                .set_arg(&original_rgba)
                .set_arg(&oklab.a)
                .set_arg(&(pixel_count as i32))
                .set_global_work_size(pixel_count)
                .enqueue_nd_range(&runtime.queue)?;
        }

        runtime.queue.finish()?;

        Ok(Self {
            width,
            height,
            original_rgba,
            ab_buffers: oklab,
        })
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
