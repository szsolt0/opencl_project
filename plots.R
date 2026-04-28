library(ggplot2)

read_bench_file <- function(path, fallback_label) {
  lines <- readLines(path, warn = FALSE)

  device_line <- lines[grepl("^Using .* device:", lines)]

  device <- if (length(device_line) > 0) {
    sub("^Using .* device: ", "", device_line[1])
  } else {
    fallback_label
  }

  bench_lines <- lines[grepl("^bench: ", lines)]

  # Keep only:
  # - GPU/OpenCL kernel lines: bench: kernel ...
  # - CPU conversion lines: bench: cpu ...
  # - filter lines: bench: HueShift { ... }: ...
  # - full runtime: bench: FULL TIME: ...
  keep <- grepl("^bench: kernel ", bench_lines) |
    grepl("^bench: cpu ", bench_lines) |
    grepl("^bench: FULL TIME:", bench_lines) |
    grepl("^bench: [A-Za-z]+( \\{.*\\})?:", bench_lines)

  bench_lines <- bench_lines[keep]

  name <- sub("^bench: ", "", bench_lines)
  name <- sub(": [0-9.]+ ?ms?$", "", name)

  # Normalize prefixes so CPU/GPU names match
  name <- sub("^kernel ", "", name)
  name <- sub("^cpu ", "", name)

  # Remove filter parameters, e.g.
  # HueShift { degrees: 90.0 } -> HueShift
  name <- sub(" \\{.*\\}", "", name)

  time_ms <- as.numeric(sub(".*: ([0-9.]+) ?ms?$", "\\1", bench_lines))

  type <- ifelse(name == "FULL TIME", "Full time", "Kernel/filter")

  data.frame(
    source = fallback_label,
    device = device,
    name = name,
    time_ms = time_ms,
    type = type,
    stringsAsFactors = FALSE
  )
}

cpu_data <- read_bench_file("bench_cpu.txt", "CPU")
gpu_data <- read_bench_file("bench_gpu.txt", "GPU")

data <- rbind(cpu_data, gpu_data)
data$source <- factor(data$source, levels = c("CPU", "GPU"))

speedup_data <- merge(
  cpu_data[, c("name", "time_ms")],
  gpu_data[, c("name", "time_ms")],
  by = "name",
  suffixes = c("_cpu", "_gpu")
)

speedup_data$speedup <- speedup_data$time_ms_cpu / speedup_data$time_ms_gpu

speedup_data$speedup_label <- ifelse(
  speedup_data$speedup >= 1,
  sprintf("%.2fx faster", speedup_data$speedup),
  sprintf("%.2fx slower", 1 / speedup_data$speedup)
)

print(speedup_data)

write.csv(speedup_data, "bench_speedup.csv", row.names = FALSE)

plot_data <- merge(
  data,
  speedup_data[, c("name", "speedup_label")],
  by = "name",
  all.x = TRUE
)

plot_data$source <- factor(plot_data$source, levels = c("CPU", "GPU"))

plot_data$bar_label <- ifelse(
  plot_data$source == "GPU" & !is.na(plot_data$speedup_label),
  paste0(sprintf("%.2f ms", plot_data$time_ms), "\n", plot_data$speedup_label),
  sprintf("%.2f ms", plot_data$time_ms)
)

p <- ggplot(plot_data, aes(x = name, y = time_ms, fill = source)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(
    aes(label = bar_label),
    position = position_dodge(width = 0.75),
    hjust = -0.08,
    size = 3.0
  ) +
  coord_flip() +
  facet_wrap(~ type, scales = "free_y") +
  expand_limits(y = max(plot_data$time_ms, na.rm = TRUE) * 1.35) +
  labs(
    title = "CPU vs GPU image filter benchmark",
    subtitle = paste(
      "CPU:", unique(cpu_data$device),
      "\nGPU:", unique(gpu_data$device)
    ),
    x = NULL,
    y = "Time (ms)",
    fill = NULL
  ) +
  theme_minimal(base_size = 13)

print(p)

ggsave("bench_comparison.pdf", plot = p, width = 11, height = 6.5, dpi = 150)
