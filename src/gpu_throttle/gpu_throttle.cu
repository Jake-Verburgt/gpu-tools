// gpu_throttle.cu
//
// Stress selected NVIDIA GPUs for N seconds while allocating ~80% of VRAM.
//
// Build:
//   nvcc -O3 -arch=sm_50 -std=c++17 -lineinfo -o gpu_throttle gpu_throttle.cu 
//
// Run examples:
//   ./gpu_throttle --gpus 0 --seconds 30
//   ./gpu_throttle --gpus 0,1,3 --seconds 120 --vram 0.80
//
// Notes:
// - Uses cudaMemGetInfo() and allocates a fraction of FREE memory.
// - Spawns one CPU thread per GPU; each thread sets its CUDA device and runs independently.
// - Workload is designed to sustain high SM utilization + memory traffic.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::ostringstream _oss;                                                 \
      _oss << "CUDA error: " << cudaGetErrorString(_e)                         \
           << " (" << static_cast<int>(_e) << ") at " << __FILE__ << ":"       \
           << __LINE__;                                                       \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

static inline bool starts_with(const std::string& s, const char* pfx) {
  return s.rfind(pfx, 0) == 0;
}

static std::vector<int> parse_gpu_list(const std::string& csv) {
  std::vector<int> ids;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    // trim spaces
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch){ return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), item.end());
    if (item.empty()) continue;
    char* endp = nullptr;
    long v = std::strtol(item.c_str(), &endp, 10);
    if (!endp || *endp != '\0') {
      throw std::runtime_error("Invalid GPU id in list: '" + item + "'");
    }
    if (v < 0 || v > 1024) {
      throw std::runtime_error("GPU id out of range: '" + item + "'");
    }
    ids.push_back(static_cast<int>(v));
  }
  // dedupe
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

// Kernel does:
// - global memory reads/writes across big buffers
// - lots of FMA to keep SMs busy
// - simple index scrambling to reduce trivial caching
__global__ void stress_kernel(float* a, float* b, float* c, size_t n, uint32_t iters) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  
  // A tiny LCG scramble based on tid
  uint32_t x = (uint32_t)(tid * 1664525u + 1013904223u);

  for (uint32_t it = 0; it < iters; ++it) {
    for (size_t i = tid; i < n; i += stride) {
      // scramble index a bit
      x = x * 1664525u + 1013904223u;
      size_t j = (i + (x & 0xFFFFu)) % n;

      float av = a[j];
      float bv = b[(j + 1315423911u) % n];

      // Compute-heavy inner loop
      float r = av;
#pragma unroll 8
      for (int k = 0; k < 32; ++k) {
        r = fmaf(r, 1.0000001f, bv);
        r = fmaf(r, 0.9999999f, av);
      }

      c[j] = r;
      // keep inputs changing to avoid compiler being too clever
      a[j] = r * 0.5000001f + av * 0.4999999f;
      b[j] = r * 0.2500001f + bv * 0.7499999f;
    }
  }
}

struct Options {
  std::vector<int> gpus;
  int seconds = 0;
  double vram_frac = 0.80;   // fraction of FREE memory to allocate
  int block = 256;
};

static void usage(const char* prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " --gpus <id0,id1,...> --seconds <N> [--vram <0.0-0.95>] [--block <threads>]\n\n"
    << "Examples:\n"
    << "  " << prog << " --gpus 0 --seconds 30\n"
    << "  " << prog << " --gpus 0,1,2 --seconds 120 --vram 0.80\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  if (argc < 2) {
    usage(argv[0]);
    throw std::runtime_error("Missing arguments.");
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return std::string(argv[++i]);
    };

    if (arg == "--gpus") {
      opt.gpus = parse_gpu_list(need_value("--gpus"));
    } else if (arg == "--seconds" || arg == "-t") {
      opt.seconds = std::stoi(need_value("--seconds"));
    } else if (arg == "--vram") {
      opt.vram_frac = std::stod(need_value("--vram"));
    } else if (arg == "--block") {
      opt.block = std::stoi(need_value("--block"));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opt.gpus.empty()) throw std::runtime_error("No GPUs specified. Use --gpus 0,1,...");
  if (opt.seconds <= 0) throw std::runtime_error("--seconds must be > 0");
  if (!(opt.vram_frac > 0.0 && opt.vram_frac < 0.96)) throw std::runtime_error("--vram should be between (0.0, 0.95]");
  if (opt.block <= 0 || opt.block > 1024) throw std::runtime_error("--block must be in 1..1024");

  return opt;
}

static void gpu_worker(int dev, int seconds, double vram_frac, int block, std::atomic<bool>& any_fail) {
  try {
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Create a stream so we don't serialize on default stream across host actions
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    // Allocate a fraction of FREE memory, but leave a little headroom for runtime overhead.
    const size_t headroom = 256ull * 1024ull * 1024ull; // 256 MiB
    size_t usable_free = (free_b > headroom) ? (free_b - headroom) : (free_b / 2);
    size_t target = static_cast<size_t>(usable_free * vram_frac);

    // We'll split target across 3 buffers a,b,c of floats.
    // Ensure size is multiple of sizeof(float).
    size_t bytes_each = (target / 3);
    bytes_each -= (bytes_each % sizeof(float));
    if (bytes_each < 64ull * 1024ull * 1024ull) {
      throw std::runtime_error("Not enough free VRAM on device to allocate meaningful buffers.");
    }

    size_t n = bytes_each / sizeof(float);

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes_each));
    CUDA_CHECK(cudaMalloc(&b, bytes_each));
    CUDA_CHECK(cudaMalloc(&c, bytes_each));

    // Initialize buffers (device memset + one kernel warmup is fine)
    CUDA_CHECK(cudaMemsetAsync(a, 0x3f, bytes_each, stream)); // ~0.74f-ish bit patterns
    CUDA_CHECK(cudaMemsetAsync(b, 0x3e, bytes_each, stream));
    CUDA_CHECK(cudaMemsetAsync(c, 0x00, bytes_each, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Choose a big grid. Cap blocks to avoid ridiculous launch overhead.
    int sm = prop.multiProcessorCount;
    int blocks = sm * 20; // heuristic
    blocks = std::min(blocks, 65535); // 1D grid cap in x for older models

    // Control kernel "duration per launch" via iters.
    // Larger iters reduces host overhead; adjust as desired.
    uint32_t iters = 2;

    // Warmup
    stress_kernel<<<blocks, block, 0, stream>>>(a, b, c, n, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t0 = std::chrono::steady_clock::now();
    auto tend = t0 + std::chrono::seconds(seconds);

    std::cout
      << "[GPU " << dev << " | " << prop.name << "] "
      << "Alloc each=" << (bytes_each / (1024.0 * 1024.0)) << " MiB (x3), "
      << "free=" << (free_b / (1024.0 * 1024.0)) << " MiB, total=" << (total_b / (1024.0 * 1024.0)) << " MiB, "
      << "running " << seconds << "s...\n";

    while (std::chrono::steady_clock::now() < tend) {
      stress_kernel<<<blocks, block, 0, stream>>>(a, b, c, n, iters);
      CUDA_CHECK(cudaGetLastError());
      // Synchronize to keep it "honest" and sustain continuous load
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "[GPU " << dev << "] done.\n";
  } catch (const std::exception& e) {
    any_fail.store(true);
    std::cerr << "[GPU " << dev << "] ERROR: " << e.what() << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      std::cerr << "No CUDA devices found.\n";
      return 2;
    }

    for (int d : opt.gpus) {
      if (d < 0 || d >= device_count) {
        std::ostringstream oss;
        oss << "Requested GPU " << d << " but system has " << device_count << " devices.";
        throw std::runtime_error(oss.str());
      }
    }

    std::atomic<bool> any_fail{false};
    std::vector<std::thread> threads;
    threads.reserve(opt.gpus.size());

    for (int dev : opt.gpus) {
      threads.emplace_back(gpu_worker, dev, opt.seconds, opt.vram_frac, opt.block, std::ref(any_fail));
    }

    for (auto& t : threads) t.join();

    return any_fail.load() ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 2;
  }
}
