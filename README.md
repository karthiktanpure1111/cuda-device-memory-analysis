# GPU Batch Image Box Filter (CUDA Streams)

Coursera GPU Programming Specialization capstone: batch box-filter of grayscale images on the GPU using CUDA streams, events, and a two-pass separable kernel.

## Overview

The program:

1. Reads every `.pgm` image in an input directory.
2. Copies each image to the device on a rotating pool of CUDA streams.
3. Applies a separable box filter (row pass + column pass).
4. Copies results back and writes `filtered_<name>.pgm`.
5. Logs per-image GPU time to stdout and `data/output/timings.csv`.

This uses course topics: multi-stream asynchronous copies, CUDA events for timing, and image processing at scale (many images in one run).

## Requirements

- NVIDIA GPU + CUDA Toolkit (`nvcc` on PATH)
- Linux, g++ compatible with the installed CUDA
- Python 3 + Pillow + NumPy only to generate sample input images

## Build

```bash
make
```

Produces `bin/box_filter`.

## Run

```bash
# generate sample PGM inputs (once)
python3 tools/generate_inputs.py

# default: data/input -> data/output, kernel=5, streams=4
./run.sh

# or explicit CLI
./bin/box_filter --input data/input --output data/output --kernel 7 --streams 4
```

Flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input` | Directory of P5 PGM files | `data/input` |
| `--output` | Output directory | `data/output` |
| `--kernel` | Odd box size | `5` |
| `--streams` | Number of CUDA streams | `4` |

## Code layout

```
src/box_filter.cu   CUDA + host CLI
tools/generate_inputs.py
Makefile
run.sh
data/input/         sample PGM images
data/output/        filtered images, timings.csv, run.log
bin/                executable
```

## Algorithm

Separable box filter of odd width `K = 2r+1`:

- `BoxRow`: average `K` neighbors on the row (clamp at borders)
- `BoxCol`: average `K` neighbors on the column

Work is launched as 16x16 thread blocks. Streams overlap H2D / compute / D2H across images. `cudaEventElapsedTime` records per-image GPU time.

## Sample results

See `artifacts/` for before/after PNG previews, `timings.csv`, and `run.log`. After you run on a real GPU, replace those files with your machine's output so reviewers see your device name and times.

## Next steps

- Pin host buffers (`cudaMallocHost`) for faster async copies
- Multi-GPU split of the file list
- Swap the custom kernel for NPP `nppiFilterBox_8u_C1R`
