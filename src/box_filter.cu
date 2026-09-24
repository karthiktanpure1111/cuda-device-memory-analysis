// Batch box filter on grayscale images using CUDA streams.
// CLI: ./bin/box_filter --input DIR --output DIR --kernel N --streams S
//
// Build: make
// Style: Google C++ with CUDA extensions.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cctype>

namespace {

constexpr int kMaxPath = 1024;

void CheckCuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << file
              << ":" << line << std::endl;
    std::exit(1);
  }
}
#define CHECK_CUDA(e) CheckCuda((e), __FILE__, __LINE__)

struct Args {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  int kernel = 5;
  int streams = 4;
};

void PrintUsage(const char* prog) {
  std::cout
      << "Usage: " << prog
      << " --input DIR --output DIR --kernel N --streams S\n"
      << "  --input    Input directory of PGM images (default data/input)\n"
      << "  --output   Output directory (default data/output)\n"
      << "  --kernel   Odd box-filter size (default 5)\n"
      << "  --streams  CUDA streams to overlap work (default 4)\n";
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--help" || k == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << k << "\n";
      PrintUsage(argv[0]);
      std::exit(1);
    }
    std::string v = argv[++i];
    if (k == "--input") {
      a.input_dir = v;
    } else if (k == "--output") {
      a.output_dir = v;
    } else if (k == "--kernel") {
      a.kernel = std::atoi(v.c_str());
    } else if (k == "--streams") {
      a.streams = std::atoi(v.c_str());
    } else {
      std::cerr << "Unknown flag " << k << "\n";
      PrintUsage(argv[0]);
      std::exit(1);
    }
  }
  if (a.kernel < 1 || a.kernel % 2 == 0) {
    std::cerr << "kernel must be a positive odd integer\n";
    std::exit(1);
  }
  if (a.streams < 1) a.streams = 1;
  return a;
}

// Minimal P5 PGM loader/saver (grayscale 8-bit).
bool LoadPgm(const std::string& path, std::vector<unsigned char>* pixels,
             int* w, int* h) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[16];
  if (std::fscanf(f, "%15s", magic) != 1 || std::strcmp(magic, "P5") != 0) {
    std::fclose(f);
    return false;
  }
  int maxval = 0;
  if (std::fscanf(f, "%d %d %d", w, h, &maxval) != 3) {
    std::fclose(f);
    return false;
  }
  std::fgetc(f);  // consume single whitespace
  if (*w <= 0 || *h <= 0 || maxval != 255) {
    std::fclose(f);
    return false;
  }
  pixels->resize(static_cast<size_t>(*w) * static_cast<size_t>(*h));
  size_t n = std::fread(pixels->data(), 1, pixels->size(), f);
  std::fclose(f);
  return n == pixels->size();
}

bool SavePgm(const std::string& path, const unsigned char* pixels, int w,
             int h) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  std::fprintf(f, "P5\n%d %d\n255\n", w, h);
  size_t n = std::fwrite(pixels, 1, static_cast<size_t>(w) * h, f);
  std::fclose(f);
  return n == static_cast<size_t>(w) * h;
}

std::vector<std::string> ListPgm(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 4) {
      std::string ext = n.substr(n.size() - 4);
      for (char& c : ext) c = static_cast<char>(std::tolower(c));
      if (ext == ".pgm") out.push_back(dir + "/" + n);
    }
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::string Basename(const std::string& p) {
  auto s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

// Separable box filter: two passes. Shared-memory tile with halo.
__global__ void BoxRow(const unsigned char* in, float* tmp, int w, int h,
                       int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float s = 0.f;
  int n = 2 * r + 1;
  for (int k = -r; k <= r; ++k) {
    int xx = x + k;
    if (xx < 0) xx = 0;
    if (xx >= w) xx = w - 1;
    s += static_cast<float>(in[y * w + xx]);
  }
  tmp[y * w + x] = s / static_cast<float>(n);
}

__global__ void BoxCol(const float* tmp, unsigned char* out, int w, int h,
                       int r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float s = 0.f;
  int n = 2 * r + 1;
  for (int k = -r; k <= r; ++k) {
    int yy = y + k;
    if (yy < 0) yy = 0;
    if (yy >= h) yy = h - 1;
    s += tmp[yy * w + x];
  }
  float v = s / static_cast<float>(n);
  if (v < 0.f) v = 0.f;
  if (v > 255.f) v = 255.f;
  out[y * w + x] = static_cast<unsigned char>(v + 0.5f);
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  auto files = ListPgm(args.input_dir);
  if (files.empty()) {
    std::cerr << "No .pgm files in " << args.input_dir << "\n";
    return 1;
  }

  int gpu = 0;
  CHECK_CUDA(cudaGetDevice(&gpu));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu));
  std::cout << "GPU: " << prop.name << "  files=" << files.size()
            << "  kernel=" << args.kernel << "  streams=" << args.streams
            << "\n";

  std::vector<cudaStream_t> streams(args.streams);
  for (int i = 0; i < args.streams; ++i) {
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  std::string csv_path = args.output_dir + "/timings.csv";
  std::ofstream csv(csv_path);
  csv << "file,width,height,bytes,ms\n";

  auto wall0 = std::chrono::steady_clock::now();
  int radius = args.kernel / 2;

  for (size_t i = 0; i < files.size(); ++i) {
    int w = 0, h = 0;
    std::vector<unsigned char> host_in;
    if (!LoadPgm(files[i], &host_in, &w, &h)) {
      std::cerr << "Failed to load " << files[i] << "\n";
      continue;
    }
    size_t n = static_cast<size_t>(w) * h;
    cudaStream_t st = streams[i % streams.size()];

    unsigned char *d_in = nullptr, *d_out = nullptr;
    float* d_tmp = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, n));
    CHECK_CUDA(cudaMalloc(&d_out, n));
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    CHECK_CUDA(cudaEventRecord(ev0, st));

    CHECK_CUDA(cudaMemcpyAsync(d_in, host_in.data(), n, cudaMemcpyHostToDevice,
                               st));

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    BoxRow<<<grid, block, 0, st>>>(d_in, d_tmp, w, h, radius);
    BoxCol<<<grid, block, 0, st>>>(d_tmp, d_out, w, h, radius);

    std::vector<unsigned char> host_out(n);
    CHECK_CUDA(cudaMemcpyAsync(host_out.data(), d_out, n,
                               cudaMemcpyDeviceToHost, st));
    CHECK_CUDA(cudaEventRecord(ev1, st));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));

    std::string out_name =
        args.output_dir + "/filtered_" + Basename(files[i]);
    if (!SavePgm(out_name, host_out.data(), w, h)) {
      std::cerr << "Failed to write " << out_name << "\n";
    }
    csv << Basename(files[i]) << "," << w << "," << h << "," << n << "," << ms
        << "\n";
    std::cout << Basename(files[i]) << "  " << w << "x" << h << "  " << ms
              << " ms\n";

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
  }

  for (auto s : streams) CHECK_CUDA(cudaStreamDestroy(s));
  auto wall1 = std::chrono::steady_clock::now();
  double wall_ms =
      std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  std::cout << "wall_ms=" << wall_ms << "\n";
  csv << "TOTAL,,,," << wall_ms << "\n";
  csv.close();
  return 0;
}
