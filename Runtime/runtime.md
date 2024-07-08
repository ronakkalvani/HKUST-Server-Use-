Data set 1 size : 10^6
Data set 2 size : 10^6
Key mod value : 10^7

NVPROF statistics while running the code

==3780568== NVPROF is profiling process 3780568, command: PkPKJoin
==3780568== Profiling application: PkPKJoin
==3780568== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   81.52%  60.228ms         3  20.076ms  5.3462ms  27.770ms  [CUDA memcpy HtoD]
                    6.81%  5.0341ms         1  5.0341ms  5.0341ms  5.0341ms  countSplits(int*, int*, int, int)
                    5.43%  4.0122ms         1  4.0122ms  4.0122ms  4.0122ms  BlockSortKernel2(int*, int*, int*, int, int)
                    1.80%  1.3290ms         1  1.3290ms  1.3290ms  1.3290ms  initCurand(curandStateXORWOW*, unsigned long, int)
                    1.64%  1.2123ms         1  1.2123ms  1.2123ms  1.2123ms  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int*, int*, cub::ScanTileState<int, bool=1>, cub::Sum, cub::detail::InputValue<int, int*>, int, int>(cub::DeviceScanPolicy<int>::Policy600, int*, int*, int, int, bool=1, cub::ScanTileState<int, bool=1>)
                    0.99%  733.58us         1  733.58us  733.58us  733.58us  BlockSortKernel(int*, int*, int)
                    0.88%  651.36us         3  217.12us  20.833us  573.70us  [CUDA memset]
                    0.24%  175.08us         1  175.08us  175.08us  175.08us  segmentedPrefixSum(int*, int*, int, int)
                    0.17%  126.06us         1  126.06us  126.06us  126.06us  JoinKernel(int*, int*, int, int*, int*)
                    0.15%  111.69us         1  111.69us  111.69us  111.69us  Assign(int*, int*, int*, int*, int*, int, int)
                    0.15%  107.30us         1  107.30us  107.30us  107.30us  findSplitsKernel(int const *, int*, int const *, int, int)
                    0.07%  50.658us         1  50.658us  50.658us  50.658us  sampleElements(curandStateXORWOW*, int*, int*, int, int)
                    0.03%  24.291us         5  4.8580us  4.7680us  5.0570us  void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, unsigned int>(cub::NullType*, int)
                    0.03%  19.873us         3  6.6240us  6.4970us  6.7200us  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>*, bool=1 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, cub::NullType, unsigned int>**>)
                    0.02%  15.553us         2  7.7760us  3.1040us  12.449us  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>*, bool=0 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, cub::NullType, unsigned int>**>)
                    0.02%  15.073us         1  15.073us  15.073us  15.073us  partitions(int*, int*, int)
                    0.02%  14.305us         1  14.305us  14.305us  14.305us  Splitterss(int*, int*, int, int)
                    0.02%  12.449us         3  4.1490us  3.8400us  4.6090us  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=1, bool=0, int, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, bool=1*, cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *>)
                    0.01%  10.464us         2  5.2320us  5.0880us  5.3760us  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800, bool=0, bool=0, int, unsigned int>(cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, bool=0*, cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, cub::NullType, unsigned int>::Policy800 const *>)
                    0.00%  2.0480us         1  2.0480us  2.0480us  2.0480us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
      API calls:   87.31%  2.46724s        28  88.116ms  4.2570us  2.46690s  cudaLaunchKernel
                    9.62%  271.89ms        17  15.993ms  6.6030us  269.35ms  cudaMalloc
                    2.26%  63.998ms         3  21.333ms  4.5284ms  31.168ms  cudaMemcpy
                    0.38%  10.728ms        12  894.02us  4.5960us  6.5900ms  cudaFree
                    0.36%  10.062ms      1140  8.8250us     138ns  2.0537ms  cuDeviceGetAttribute
                    0.06%  1.6535ms         1  1.6535ms  1.6535ms  1.6535ms  cudaDeviceSynchronize
                    0.00%  61.739us        10  6.1730us  3.9980us  13.971us  cuDeviceGetName
                    0.00%  50.326us         3  16.775us  12.975us  23.001us  cudaMemset
                    0.00%  26.165us        10  2.6160us  1.3040us  7.7320us  cuDeviceGetPCIBusId
                    0.00%  23.948us        13  1.8420us     470ns  10.296us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  19.705us        98     201ns     133ns  5.0650us  cudaGetLastError
                    0.00%  14.943us         1  14.943us  14.943us  14.943us  cudaFuncGetAttributes
                    0.00%  5.5270us        34     162ns     131ns     283ns  cudaPeekAtLastError
                    0.00%  5.1950us         9     577ns     326ns  1.5630us  cudaGetDevice
                    0.00%  3.9940us        20     199ns     141ns     660ns  cuDeviceGet
                    0.00%  3.3560us        10     335ns     279ns     580ns  cuDeviceTotalMem
                    0.00%  2.9580us         3     986ns     480ns  1.5990us  cudaDeviceGetAttribute
                    0.00%  2.7010us         3     900ns     291ns  1.8200us  cuDeviceGetCount
                    0.00%  2.3520us        10     235ns     214ns     286ns  cuDeviceGetUuid
                    0.00%     367ns         1     367ns     367ns     367ns  cuModuleGetLoadingMode
                    0.00%     217ns         1     217ns     217ns     217ns  cudaGetDeviceCount
