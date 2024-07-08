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
