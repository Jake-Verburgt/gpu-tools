# gpu-tools
A Collection of tools for charactierizing GPU capabilities and metrics




## GPU Throttle

Manual Compilation
```
nvcc -O3 -std=c++17 -lineinfo -o gpu_throttle gpu_throttle.cu
./gpu_throttle --gpus 0,1 --seconds 60 --vram 0.80

```