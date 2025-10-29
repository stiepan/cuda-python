#ifndef CUDA_CORE_LAYOUT_HPP
#define CUDA_CORE_LAYOUT_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#define STRIDED_LAYOUT_MAX_NDIM 32
#define AXIS_MASK_ALL 0xFFFFFFFF

inline int64_t c_abs(int64_t x)
{
    return std::abs(x);
}

template <typename T>
void swap(std::vector<T> &a, std::vector<T> &b) noexcept
{
    std::swap(a, b);
}

inline void order_from_strides(std::vector<int> &indices, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides)
{
    int ndim = shape.size();
    indices.resize(ndim);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&strides, &shape](int i, int j)
              {
                  int64_t stride_i = c_abs(strides[i]);
                  int64_t stride_j = c_abs(strides[j]);
                  if (stride_i != stride_j)
                  {
                      return stride_i > stride_j;
                  }
                  int64_t shape_i = shape[i];
                  int64_t shape_j = shape[j];
                  if (shape_i != shape_j)
                  {
                      return shape_i > shape_j;
                  }
                  return i < j;
              });
}


inline void resize(std::vector<int64_t> &shape, std::vector<int64_t> &strides, int ndim)
{
    shape.resize(ndim);
    strides.resize(ndim);
}

#endif // CUDA_CORE_LAYOUT_HPP