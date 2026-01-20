// nccl_throttle.cu
//
// Single-node multi-GPU NCCL AllReduce stress/throttle:
// - allocate ~vram_frac of FREE memory on each selected GPU
// - repeatedly run in-place ncclAllReduce(SUM) for N seconds
//
// Build (paths may vary):
//   nvcc -O3 -std=c++17 -arch=sm_50 -lineinfo -o nccl_throttle nccl_throttle.cu -lnccl
//
// Run:
//   ./nccl_allreduce_throttle --gpus 0,1 --seconds 60 --vram 0.80
//
// Useful envs (optional):
//   NCCL_DEBUG=INFO
//   NCCL_P2P_LEVEL=NVL   (or PIX, SYS, etc.)
//   NCCL_TOPO_DUMP_FILE=topo.xml

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call) do {                                    \
  cudaError_t e = (call);                                        \
  if (e != cudaSuccess) {                                        \
    std::ostringstream oss;                                      \
    oss << "CUDA error: " << cudaGetErrorString(e)               \
        << " (" << (int)e << ") at " << __FILE__ << ":" << __LINE__; \
    throw std::runtime_error(oss.str());                         \
  }                                                              \
} while (0)

#define NCCL_CHECK(call) do {                                    \
  ncclResult_t r = (call);                                       \
  if (r != ncclSuccess) {                                        \
    std::ostringstream oss;                                      \
    oss << "NCCL error: " << ncclGetErrorString(r)               \
        << " (" << (int)r << ") at " << __FILE__ << ":" << __LINE__; \
    throw std::runtime_error(oss.str());                         \
  }                                                              \
} while (0)

struct Barrier {
  explicit Barrier(int count) : count_(count), waiting_(0), phase_(0) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    int my_phase = phase_;
    if (++waiting_ == count_) {
      waiting_ = 0;
      phase_++;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&]{ return phase_ != my_phase; });
    }
  }
private:
  int count_;
  int waiting_;
  int phase_;
  std::mutex m_;
  std::condition_variable cv_;
};

static std::vector<int> parse_gpu_list(const std::string& csv) {
  std::vector<int> ids;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // trim
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch){ return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), item.end());
    if (item.empty()) continue;
    char* endp = nullptr;
    long v = std::strtol(item.c_str(), &endp, 10);
    if (!endp || *endp != '\0') throw std::runtime_error("Invalid GPU id: '" + item + "'");
    if (v < 0 || v > 1024) throw std::runtime_error("GPU id out of range: '" + item + "'");
    ids.push_back((int)v);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

struct Options {
  std::vector<int> gpus;
  int seconds = 0;
  double vram_frac = 0.80;     // fraction of FREE VRAM to allocate total (buffer+ballast)
  double payload_frac = 0.60;  // fraction of allocated bytes used for the AllReduce payload buffer
};

static void usage(const char* prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " --gpus <id0,id1,...> --seconds <N> [--vram <0.0-0.95>] [--payload <0.1-0.95>]\n\n"
    << "Examples:\n"
    << "  " << prog << " --gpus 0,1 --seconds 60\n"
    << "  " << prog << " --gpus 0,1,2,3 --seconds 120 --vram 0.80 --payload 0.70\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  if (argc < 2) { usage(argv[0]); throw std::runtime_error("Missing args."); }
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name)->std::string {
      if (i+1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (a == "--gpus") opt.gpus = parse_gpu_list(need("--gpus"));
    else if (a == "--seconds" || a == "-t") opt.seconds = std::stoi(need("--seconds"));
    else if (a == "--vram") opt.vram_frac = std::stod(need("--vram"));
    else if (a == "--payload") opt.payload_frac = std::stod(need("--payload"));
    else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else { usage(argv[0]); throw std::runtime_error("Unknown arg: " + a); }
  }
  if (opt.gpus.empty()) throw std::runtime_error("No GPUs specified.");
  if (opt.seconds <= 0) throw std::runtime_error("--seconds must be > 0");
  if (!(opt.vram_frac > 0.0 && opt.vram_frac <= 0.95)) throw std::runtime_error("--vram must be in (0, 0.95]");
  if (!(opt.payload_frac > 0.1 && opt.payload_frac <= 0.95)) throw std::runtime_error("--payload must be in (0.1, 0.95]");
  return opt;
}

struct RankState {
  int dev = -1;
  int rank = -1;
  int nranks = 0;
  ncclComm_t comm{};
  cudaStream_t stream{};
  float* payload = nullptr;    // used for AllReduce
  void* ballast = nullptr;     // extra allocation to hold VRAM
  size_t payload_bytes = 0;
  size_t ballast_bytes = 0;
};

__global__ void init_kernel(float* p, size_t n, float val) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += stride) p[i] = val;
}

static void rank_worker(const Options& opt,
                        RankState& st,
                        ncclUniqueId id,
                        Barrier& barrier,
                        std::atomic<bool>& any_fail) {
  try {
    CUDA_CHECK(cudaSetDevice(st.dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, st.dev));

    CUDA_CHECK(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking));
    NCCL_CHECK(ncclCommInitRank(&st.comm, st.nranks, id, st.rank));

    // Memory sizing based on FREE memory to avoid OOM if GPU is already used
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    const size_t headroom = 512ull * 1024ull * 1024ull; // 512 MiB
    size_t usable_free = (free_b > headroom) ? (free_b - headroom) : (free_b / 2);
    size_t target_alloc = (size_t)(usable_free * opt.vram_frac);

    // Split target into payload (AllReduce buffer) + ballast (just to occupy VRAM)
    size_t payload_bytes = (size_t)(target_alloc * opt.payload_frac);
    payload_bytes -= payload_bytes % sizeof(float);
    // Clamp payload to something reasonable if extremely small
    if (payload_bytes < 64ull * 1024ull * 1024ull) payload_bytes = 64ull * 1024ull * 1024ull;

    size_t ballast_bytes = (target_alloc > payload_bytes) ? (target_alloc - payload_bytes) : 0;
    // Ballast alignment
    ballast_bytes -= ballast_bytes % 256;

    st.payload_bytes = payload_bytes;
    st.ballast_bytes = ballast_bytes;

    CUDA_CHECK(cudaMalloc(&st.payload, st.payload_bytes));
    if (st.ballast_bytes >= (32ull * 1024ull * 1024ull)) {
      CUDA_CHECK(cudaMalloc(&st.ballast, st.ballast_bytes));
      // Touch ballast a bit so it’s actually backed
      CUDA_CHECK(cudaMemsetAsync(st.ballast, 0, st.ballast_bytes, st.stream));
    }

    // Initialize payload
    size_t n = st.payload_bytes / sizeof(float);
    int block = 256;
    int grid = (int)std::min<size_t>((n + block - 1) / block, 65535);
    init_kernel<<<grid, block, 0, st.stream>>>(st.payload, n, 1.0f + st.rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st.stream));

    if (st.rank == 0) {
      std::cout << "Ranks: " << st.nranks << " (single-node)\n";
    }
    std::cout
      << "[rank " << st.rank << " | GPU " << st.dev << " | " << prop.name << "] "
      << "free=" << (free_b / (1024.0*1024.0)) << " MiB, total=" << (total_b / (1024.0*1024.0)) << " MiB, "
      << "payload=" << (st.payload_bytes / (1024.0*1024.0)) << " MiB, "
      << "ballast=" << (st.ballast_bytes / (1024.0*1024.0)) << " MiB\n";

    // Sync before starting the timed loop
    barrier.wait();

    auto t0 = std::chrono::steady_clock::now();
    auto tend = t0 + std::chrono::seconds(opt.seconds);

    // Chunking: operating on the entire payload each iteration gives the most “clean” bandwidth load.
    // If you want to reduce per-iteration latency, you could do smaller chunks more frequently.
    const size_t count = st.payload_bytes / sizeof(float);

    // Main throttle loop
    while (std::chrono::steady_clock::now() < tend) {
      // Keep ranks approximately in lock-step to avoid stragglers
      barrier.wait();

      NCCL_CHECK(ncclAllReduce(
          (const void*)st.payload,
          (void*)st.payload,          // in-place
          (size_t)count,
          ncclFloat,
          ncclSum,
          st.comm,
          st.stream));

      // Make it "honest": wait for this collective to complete before next iteration
      CUDA_CHECK(cudaStreamSynchronize(st.stream));
    }

    barrier.wait();

    // Cleanup
    if (st.ballast) CUDA_CHECK(cudaFree(st.ballast));
    if (st.payload) CUDA_CHECK(cudaFree(st.payload));
    NCCL_CHECK(ncclCommDestroy(st.comm));
    CUDA_CHECK(cudaStreamDestroy(st.stream));

    std::cout << "[rank " << st.rank << " | GPU " << st.dev << "] done.\n";
  } catch (const std::exception& e) {
    any_fail.store(true);
    std::cerr << "[rank " << st.rank << " | GPU " << st.dev << "] ERROR: " << e.what() << "\n";
    // Best-effort cleanup on failure
    try {
      if (st.ballast) cudaFree(st.ballast);
      if (st.payload) cudaFree(st.payload);
      if (st.comm) ncclCommDestroy(st.comm);
      if (st.stream) cudaStreamDestroy(st.stream);
    } catch (...) {}
  }
}

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("No CUDA devices found.");

    for (int d : opt.gpus) {
      if (d < 0 || d >= device_count) {
        std::ostringstream oss;
        oss << "Requested GPU " << d << " but system has " << device_count << " devices.";
        throw std::runtime_error(oss.str());
      }
    }

    int nranks = (int)opt.gpus.size();
    std::vector<RankState> states(nranks);
    for (int r = 0; r < nranks; ++r) {
      states[r].rank = r;
      states[r].nranks = nranks;
      states[r].dev = opt.gpus[r];
    }

    ncclUniqueId id{};
    NCCL_CHECK(ncclGetUniqueId(&id));

    Barrier barrier(nranks);
    std::atomic<bool> any_fail{false};

    std::vector<std::thread> threads;
    threads.reserve(nranks);

    for (int r = 0; r < nranks; ++r) {
      threads.emplace_back(rank_worker, std::cref(opt), std::ref(states[r]), id,
                           std::ref(barrier), std::ref(any_fail));
    }
    for (auto& t : threads) t.join();

    return any_fail.load() ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 2;
  }
}
