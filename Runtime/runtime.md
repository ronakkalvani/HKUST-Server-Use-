Data set 1 size : 10^6
Data set 2 size : 10^6
Key mod value : 10^8

NVPROF statistics while running the code on randomly generated data set

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

Crystal library statistics on same data set

==3786656== NVPROF is profiling process 3786656, command: anyfile
{"time_memset":0.031168,"time_build":296.066,"time_probe":0.211168}
{"num_dim":1000000,"num_fact":1000000,"radix":0,"time_partition_build":0,"time_partition_probe":0,"time_partition_total":0,"time_build":296.066,"time_probe":0.211168,"time_extra":0.031168,"time_join_total":296.309}
==3786656== Profiling application: anyfile
==3786656== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   91.26%  10.717ms         4  2.6793ms  2.6776ms  2.6802ms  [CUDA memcpy HtoD]
                    6.90%  810.25us         1  810.25us  810.25us  810.25us  void build_kernel<int=128, int=4>(int*, int*, int, int*, int)
                    1.68%  197.80us         1  197.80us  197.80us  197.80us  void probe_kernel<int=128, int=4>(int*, int*, int, int*, int, __int64*)
                    0.16%  18.593us         2  9.2960us  1.4400us  17.153us  [CUDA memset]
      API calls:   50.83%  295.28ms         2  147.64ms  11.869us  295.27ms  cudaLaunchKernel
                   45.27%  263.00ms         6  43.834ms  141.28us  261.81ms  cudaMalloc
                    1.74%  10.124ms         4  2.5310ms  1.9767ms  2.8489ms  cudaMemcpy
                    1.68%  9.7874ms      1140  8.5850us     137ns  2.0829ms  cuDeviceGetAttribute
                    0.23%  1.3461ms         4  336.52us  6.5660us  801.21us  cudaEventSynchronize
                    0.18%  1.0747ms         5  214.94us  179.62us  265.58us  cudaFree
                    0.01%  77.355us         2  38.677us  5.1440us  72.211us  cudaMemset
                    0.01%  54.811us        10  5.4810us  3.6430us  13.683us  cuDeviceGetName
                    0.01%  35.246us         9  3.9160us  2.1610us  8.3800us  cudaEventRecord
                    0.00%  26.509us        10  2.6500us  1.2820us  7.0470us  cuDeviceGetPCIBusId
                    0.00%  25.217us         2  12.608us  1.3310us  23.886us  cudaEventCreate
                    0.00%  24.596us         6  4.0990us  1.5690us  11.039us  cudaEventCreateWithFlags
                    0.00%  23.406us        12  1.9500us     489ns  9.1270us  cudaGetDevice
                    0.00%  16.366us        51     320ns     131ns  6.1930us  cudaGetLastError
                    0.00%  6.7070us         4  1.6760us     978ns  2.4250us  cudaEventElapsedTime
                    0.00%  6.5960us         5  1.3190us     958ns  1.8130us  cudaEventDestroy
                    0.00%  3.8810us        20     194ns     143ns     673ns  cuDeviceGet
                    0.00%  3.2820us        10     328ns     269ns     577ns  cuDeviceTotalMem
                    0.00%  2.7340us         3     911ns     231ns  1.8670us  cuDeviceGetCount
                    0.00%  2.1820us        10     218ns     189ns     296ns  cuDeviceGetUuid
                    0.00%     355ns         1     355ns     355ns     355ns  cuModuleGetLoadingMode


