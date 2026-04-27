use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::context::Context;
use opencl3::Result;

use std::ptr;

pub struct AbBuffers<T> {
    pub a: Buffer<T>,
    pub b: Buffer<T>,
    pub a_is_current: bool,
}

impl<T> AbBuffers<T> {
    pub fn create(size: usize, context: &Context) -> Result<Self> {
        let a = unsafe {
            Buffer::<T>::create(
                context,
                CL_MEM_READ_WRITE,
                size,
                ptr::null_mut(),
            )?
        };

        let b = unsafe {
            Buffer::<T>::create(
                context,
                CL_MEM_READ_WRITE,
                size,
                ptr::null_mut(),
            )?
        };

        Ok(Self {a, b, a_is_current: true})
    }

    pub fn current(&self) -> &Buffer<T> {
        if self.a_is_current { &self.a } else { &self.b }
    }

    pub fn other(&self) -> &Buffer<T> {
        if self.a_is_current { &self.b } else { &self.a }
    }

    pub fn swap(&mut self) {
        self.a_is_current = !self.a_is_current;
    }
}
