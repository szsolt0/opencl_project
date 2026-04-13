use opencl3::memory::Buffer;

pub struct AbBuffers<T> {
    pub a: Buffer<T>,
    pub b: Buffer<T>,
    pub a_is_current: bool,
}

impl<T> AbBuffers<T> {
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
