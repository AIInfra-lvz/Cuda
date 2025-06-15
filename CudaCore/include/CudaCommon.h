#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename _T>
void check(_T result, char const *const func, const char *const file, int const line) 
{
  if (result) 
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    return;
  }
}