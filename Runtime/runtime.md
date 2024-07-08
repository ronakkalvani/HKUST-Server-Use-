Data set 1 size : 10^6
Data set 2 size : 10^6
Key mod value : 10^8

NVPROF statistics while running the code

==3785409== NVPROF is profiling process 3785409, command: PKPKJOIN
==3785409== Profiling application: PKPKJOIN
==3785409== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.44%  534.81ms         3  178.27ms  5.2992ms  264.77ms  [CUDA memcpy HtoD]
                    0.89%  4.8801ms         1  4.8801ms  4.8801ms  4.8801ms  countSplits(int*, int*, int, int)
                    0.73%  4.0189ms         1  4.0189ms  4.0189ms  4.0189ms  BlockSortKernel2(int*, int*, int*, int, int)
                    0.23%  1.2761ms         1  1.2761ms  1.2761ms  1.2761ms  initCurand(curandStateXORWOW*, unsigned long, int)
                    0.22%  1.2142ms         1  1.2142ms  1.2142ms  1.2142ms  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int*, int*, cub::ScanTileState<int, bool=1>, cub::Sum, cub::detail::InputValue<int, int*>, int, int>(cub::DeviceScanPolicy<int>::Policy600, int*, int*, int, int, bool=1, cub::ScanTileState<int, bool=1>)
                    0.14%  792.56us         1  792.56us  792.56us  792.56us  Assign(int*, int*, int*, int*, int*, int, int)
                    0.13%  733.74us         1  733.74us  733.74us  733.74us  BlockSortKernel(int*, int*, int)
                    0.12%  652.90us         3  217.63us  20.833us  575.26us  [CUDA memset]
                    0.02%  136.07us         1  136.07us  136.07us  136.07us  segmentedPrefixSum(int*, int*, int, int)
                    0.02%  110.85us         1  110.85us  110.85us  110.85us  findSplitsKernel(int const *, int*, int const *, int, int)
                    0.01%  50.403us         1  50.403us  50.403us  50.403us  sampleElements(curandStateXORWOW*, int*, int*, int, int)
                    0.01%  42.819us         1  42.819us  42.819us  42.819us  JoinKernel(int*, int*, int, int*, int*)
                    0.00%  24.290us         5  4.8580us  4.7360us  5.1210us  void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, unsigned int>(cub::NullType*, int)
                    0.00%  21.538us         2  10.769us  9.0250us  12.513us  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>*, bool=0 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**>)
                    0.00%  19.745us         3  6.5810us  6.4640us  6.6560us  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>*, bool=1 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**>)
                    0.00%  15.009us         1  15.009us  15.009us  15.009us  partitions(int*, int*, int)
                    0.00%  14.305us         1  14.305us  14.305us  14.305us  Splitterss(int*, int*, int, int)
                    0.00%  12.513us         3  4.1710us  3.9040us  4.6080us  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, bool=1*, cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *>)
                    0.00%  10.464us         2  5.2320us  5.1200us  5.3440us  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, bool=0*, cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *>)
                    0.00%  2.0800us         1  2.0800us  2.0800us  2.0800us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
      API calls:   75.57%  2.55572s        28  91.276ms  3.9860us  2.55544s  cudaLaunchKernel
                   15.89%  537.32ms         3  179.11ms  4.4682ms  268.42ms  cudaMemcpy
                    7.84%  265.19ms        17  15.599ms  6.0940us  262.82ms  cudaMalloc
                    0.36%  12.085ms        12  1.0071ms  9.4470us  6.4133ms  cudaFree
                    0.29%  9.8906ms      1140  8.6750us     159ns  2.0199ms  cuDeviceGetAttribute
                    0.05%  1.6618ms         1  1.6618ms  1.6618ms  1.6618ms  cudaDeviceSynchronize
                    0.00%  69.403us         3  23.134us  12.629us  42.077us  cudaMemset
                    0.00%  52.933us        10  5.2930us  3.5430us  13.267us  cuDeviceGetName
                    0.00%  29.748us        13  2.2880us     509ns  13.739us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  25.133us        10  2.5130us  1.3340us  7.7580us  cuDeviceGetPCIBusId
                    0.00%  15.853us         1  15.853us  15.853us  15.853us  cudaFuncGetAttributes
                    0.00%  15.261us        98     155ns     130ns     336ns  cudaGetLastError
                    0.00%  6.0110us         9     667ns     322ns  2.4390us  cudaGetDevice
                    0.00%  5.6820us        34     167ns     140ns     303ns  cudaPeekAtLastError
                    0.00%  4.9500us        20     247ns     148ns     697ns  cuDeviceGet
                    0.00%  3.6010us         3  1.2000us     495ns  2.1630us  cudaDeviceGetAttribute
                    0.00%  3.2800us        10     328ns     277ns     557ns  cuDeviceTotalMem
                    0.00%  2.2290us        10     222ns     179ns     338ns  cuDeviceGetUuid
                    0.00%  1.4420us         3     480ns     247ns     936ns  cuDeviceGetCount
                    0.00%     366ns         1     366ns     366ns     366ns  cuModuleGetLoadingMode
                    0.00%     319ns         1     319ns     319ns     319ns  cudaGetDeviceCount

