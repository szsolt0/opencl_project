# OpenCL Image Processing Project

This project applies image transformations such as hue shift, saturation,
exposure, contrast, and blur filters. The transformations are performed in the
OkLab color space, and the GPU implementation uses OpenCL.

## Usage

```
Usage:
  target/release/opencl_project <cpu|gpu> <input> <output> <filter>...

Filters:
  exposure:<ev>
  contrast:<amount>
  saturation:<amount>
  hue_shift:<degrees>
  gaussian_blur:<radius>,<sigma>
  box_blur:<radius>

Example:
  target/release/opencl_project gpu input.png output.png hue_shift:45 saturation:1.5 gaussian_blur:4,2.5
```

## Building

For best CPU performance, build the project in release mode with native CPU
optimizations enabled:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release
Benchmarking
```

## Benchmarking

The benchmark data and comparison plot can be generated with the following
commands:

```sh
target/release/opencl_project cpu test_images/img_land.jpg test_images/out.png hue:90 saturation:1.5 exposure:1.2 > bench_cpu.txt
target/release/opencl_project gpu test_images/img_land.jpg test_images/out.png hue:90 saturation:1.5 exposure:1.2 > bench_gpu.txt
Rscript plots.R
```


The generated files are:

- `bench_comparison.pdf`: visual comparison of CPU and GPU timings
- `bench_speedup.csv`: raw timing data and calculated speedups

## Benchmark Setup

The following hardware was used for testing:

- GPU: Mesa Intel(R) Iris(R) Xe Graphics (TGL GT2)
- CPU: 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz

The benchmark image had a resolution of 5068 × 2850, containing
14,443,800 pixels.

## Benchmark Results

In the benchmark, the GPU implementation was faster than the CPU implementation
in every measured stage.

The largest performance improvements were observed in:

- RGBA to OkLab conversion
- OkLab to RGBA conversion
- Hue shift

The total wall-clock runtime of the GPU version was 6.94× faster than the
CPU version.

## Conclusion

The benchmark results show that the OpenCL GPU implementation provides a clear
performance improvement over the CPU implementation for this image-processing
pipeline.
