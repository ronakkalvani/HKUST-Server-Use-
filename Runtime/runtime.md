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



Paper : A study of the fundamental performance characteristics of gpus and cpus for data

Stats :-
==308861== NVPROF is profiling process 308861, command: ./join
Using device 0: NVIDIA GeForce RTX 2080 (PTX version 520, SM750, 46 SMs, 7663 free / 7791 total MB physmem, 448.000 GB/s @ 7000000 kHz mem clock, ECC off)

{"Time_memset":0.390656,
"Time_build"20.5914,
"Time_probe":103.692}

{"num_dim":16777216,
"Num_fact":268435456,
"radix":0, 
"time_partition_build":0, 
"time_partition_probe":0, 
"time_partition_total":0,
"time_build":20.5914,
"Time_probe":103.692,
"Time_extra":0.390656
,"time_join_total":124.674}
{"time_memset":0.361344,
"Time_build"20.5603,
"Time_probe":103.684}

{"num_dim":16777216,
"Num_fact":268435456,
"radix":0,
"Time_partition_build":0,
"Time_partition_probe":0,
"Time_partition_total":0,
"Time_build":20.5603,
"Time_probe":103.684,
"Time_extra":0.361344,
"Time_join_total":124.606}

{"time_memset":0.371264,
"Time_build"20.6035,
"time_probe":103.671}
{"num_dim":16777216,
"Num_fact":268435456,
"Radix":0,"time_partition_build":0,
"Time_partition_probe":0,
"Time_partition_total":0,
"Time_build":20.6035,
"Time_probe":103.671,
"Time_extra":0.371264,
"time_join_total":124.646}

==308861== Profiling application: ./join
==308861== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name

 GPU activities:
       80.18%  1.51228s         4  378.07ms  44.494ms  711.90ms  [CUDA memcpy HtoD]
                   16.49%  311.03ms         3  103.68ms  103.67ms  103.68ms  void probe_kernel<int=128, int=4>(int*, int*, int, int*, int, __int64*)
                    3.27%  61.664ms         3  20.555ms  20.521ms  20.595ms  void build_kernel<int=128, int=4>(int*, int*, int, int*, int)
                    0.05%  1.0215ms         6  170.24us  1.6320us  357.88us  [CUDA memset]

      API calls:   
        66.66%  1.51172s         4  377.93ms  43.603ms  712.02ms  cudaMemcpy
                    16.48%  373.74ms        12  31.145ms  5.9380us  103.69ms  cudaEventSynchronize
                   11.57%  262.48ms         1  262.48ms  262.48ms  262.48ms  cudaSetDevice
                    4.17%  94.497ms         1  94.497ms  94.497ms  94.497ms  cudaFuncGetAttributes
                    0.44%  10.029ms      1140  8.7970us     142ns  2.1858ms  cuDeviceGetAttribute
                    0.40%  8.9594ms         7  1.2799ms  300.27us  2.9614ms  cudaFree
                    0.20%  4.4596ms         8  557.46us  160.11us  1.2596ms  cudaMalloc
                    0.06%  1.2906ms         1  1.2906ms  1.2906ms  1.2906ms  cudaGetDeviceProperties
                    0.01%  119.08us         6  19.846us  6.8290us  64.027us  cudaLaunchKernel
                    0.01%  116.57us         6  19.428us  4.3850us  74.226us  cudaMemset
                    0.00%  106.33us        27  3.9380us  1.8770us  13.955us  cudaEventRecord
                    0.00%  64.899us         6  10.816us  1.2780us  54.458us  cudaEventCreate
                    0.00%  61.381us         1  61.381us  61.381us  61.381us  cudaMemGetInfo
                    0.00%  50.271us        10  5.0270us  3.4130us  12.599us  cuDeviceGetName
                    0.00%  27.140us         8  3.3920us  1.8940us  11.848us  cudaEventCreateWithFlags
                    0.00%  26.737us        88     303ns     133ns  6.1860us  cudaGetLastError
                    0.00%  23.986us        22  1.0900us     350ns  7.1700us  cudaGetDevice
                    0.00%  22.681us        10  2.2680us  1.3270us  6.6830us  cuDeviceGetPCIBusId
                    0.00%  17.364us        12  1.4470us     939ns  3.1540us  cudaEventElapsedTime
                    0.00%  11.053us         7  1.5790us  1.2570us  1.9100us  cudaEventDestroy
                    0.00%  4.1150us        20     205ns     133ns     777ns  cuDeviceGet
                    0.00%  3.2410us        10     324ns     239ns     674ns  cuDeviceTotalMem
                    0.00%  2.1080us        10     210ns     179ns     275ns  cuDeviceGetUuid
                    0.00%  1.4810us         3     493ns     226ns     980ns  cuDeviceGetCount
                    0.00%  1.3260us         2     663ns     376ns     950ns  cudaGetDeviceCount
                    0.00%     349ns         1     349ns     349ns     349ns  cuModuleGetLoadingMode
