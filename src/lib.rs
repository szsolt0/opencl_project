pub mod image_state;
pub mod ab_buffers;
pub mod opencl_runtime;
pub mod image_filter;

use crate::opencl_runtime::*;
use crate::image_state::*;

pub struct ImageConverter {
    runtime: OpenClRuntime,
    image: ImageState,
}
