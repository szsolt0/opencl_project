use opencl3::context::Context;
use opencl3::command_queue::CommandQueue;
use opencl3::device::Device;
use opencl3::device::cl_device_id;
use crate::image_filter::*;

pub struct OpenClRuntime {
    pub context: Context,
    pub queue: CommandQueue,
    pub filter_kernels: ImageFilterKernels,
}

#[derive(Debug)]
pub enum OpenClInitError {
    NoPlatform,
    NoDevice,
    OpenCl(opencl3::error_codes::ClError),
    KernelInit(KernelInitError),
}

impl From<opencl3::error_codes::ClError> for OpenClInitError {
    fn from(value: opencl3::error_codes::ClError) -> Self {
        Self::OpenCl(value)
    }
}

impl From<KernelInitError> for OpenClInitError {
    fn from(value: KernelInitError) -> Self {
        Self::KernelInit(value)
    }
}

impl std::fmt::Display for OpenClInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPlatform => write!(f, "no OpenCL platform found"),
            Self::NoDevice => write!(f, "no suitable OpenCL device found"),
            Self::OpenCl(err) => write!(f, "OpenCL error: {:?}", err),
            Self::KernelInit(err) => write!(f, "kernel init error: {}", err),
        }
    }
}

impl std::error::Error for OpenClInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KernelInit(err) => Some(err),
            _ => None,
        }
    }
}

impl OpenClRuntime {
    pub fn init(device_id: cl_device_id) -> Result<Self, OpenClInitError> {
        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, 0)?;
        let filter_kernels = ImageFilterKernels::init(&context, device_id)?;

        Ok(Self {
            context,
            queue,
            filter_kernels,
        })
    }
}
