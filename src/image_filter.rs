use opencl3::context::Context;
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use opencl3::device::cl_device_id;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFilterKind {
    Exposure,
    Contrast,
    Saturation,
    HueShift,
    GaussianBlur,
    BoxBlur,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFilter {
    Exposure { ev: f32 },
    Contrast { amount: f32 },
    Saturation { amount: f32 },
    HueShift { degrees: f32 },
    GaussianBlur { radius: u32, sigma: f32 },
    BoxBlur { radius: u32 },
}

impl ImageFilter {
    pub fn kind(&self) -> ImageFilterKind {
        match self {
            Self::Exposure { .. } => ImageFilterKind::Exposure,
            Self::Contrast { .. } => ImageFilterKind::Contrast,
            Self::Saturation { .. } => ImageFilterKind::Saturation,
            Self::HueShift { .. } => ImageFilterKind::HueShift,
            Self::GaussianBlur { .. } => ImageFilterKind::GaussianBlur,
            Self::BoxBlur { .. } => ImageFilterKind::BoxBlur,
        }
    }
}

impl ImageFilterKind {
    pub fn kernel_name(self) -> &'static str {
        match self {
            Self::Exposure => "exposure",
            Self::Contrast => "contrast",
            Self::Saturation => "saturation",
            Self::HueShift => "hue_shift",
            Self::GaussianBlur => "gaussian_blur",
            Self::BoxBlur => "box_blur",
        }
    }

    pub fn all() -> [ImageFilterKind; 4] {
        [
            Self::Exposure,
            Self::Contrast,
            Self::Saturation,
            Self::HueShift,
            // Not implementet yet
            //Self::GaussianBlur,
            //Self::BoxBlur,
        ]
    }

    pub fn is_spatial(&self) -> bool {
        match self {
            Self::BoxBlur
            | Self::GaussianBlur => true,
            _ => false,
        }
    }

    pub fn is_point(&self) -> bool {
        !self.is_spatial()
    }
}

#[derive(Debug)]
pub enum KernelInitError {
    Io(std::io::Error),
    OpenCl(opencl3::error_codes::ClError),
}

impl From<std::io::Error> for KernelInitError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<opencl3::error_codes::ClError> for KernelInitError {
    fn from(value: opencl3::error_codes::ClError) -> Self {
        Self::OpenCl(value)
    }
}

impl std::fmt::Display for KernelInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::OpenCl(err) => write!(f, "OpenCL error: {:?}", err),
        }
    }
}

impl std::error::Error for KernelInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::OpenCl(_) => None,
        }
    }
}

pub struct ImageFilterKernels {
    pub program: Program,
    pub kernels: HashMap<ImageFilterKind, Kernel>,
    pub rgba_to_oklab: Kernel,
    pub oklab_to_rgba: Kernel,
}

impl ImageFilterKernels {
    pub fn init(
        context: &Context,
        _device_id: cl_device_id,
    ) -> Result<Self, KernelInitError> {
        let source = load_kernel_source_tree("kernels")?;

        let build_opt = "-cl-std=CL1.2";
        let program = Program::create_and_build_from_source(&context, &source, build_opt).unwrap();
        //program.build(&[device_id], "-O2")?;

        let mut kernels = HashMap::new();
        for kind in ImageFilterKind::all() {
            let kernel = Kernel::create(&program, kind.kernel_name())?;
            kernels.insert(kind, kernel);
        }

        let rgba_to_oklab = Kernel::create(&program, "rgba8_to_oklab")?;
        let oklab_to_rgba = Kernel::create(&program, "oklab_to_rgba8")?;

        Ok(Self { program, kernels, rgba_to_oklab, oklab_to_rgba })
    }

    pub fn kernel_for_filter(&self, filter: &ImageFilter) -> &Kernel {
        self.kernels.get(&filter.kind()).unwrap()
    }

    pub fn kernel_for_kind(&self, kind: ImageFilterKind) -> &Kernel {
        self.kernels.get(&kind).unwrap()
    }

    pub fn program(&self) -> &Program {
        &self.program
    }
}

fn load_kernel_source_tree(root: impl AsRef<Path>) -> Result<String, KernelInitError> {
    let mut files = Vec::new();
    collect_cl_files(root.as_ref(), &mut files)?;
    files.sort();

    let mut source = String::new();
    for path in files {
        let text = fs::read_to_string(&path)?;
        source.push_str(&format!("// ===== {} =====\n", path.display()));
        source.push_str(&text);
        source.push_str("\n\n");
    }

    Ok(source)
}

fn collect_cl_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_cl_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("cl") {
            out.push(path);
        }
    }

    Ok(())
}
