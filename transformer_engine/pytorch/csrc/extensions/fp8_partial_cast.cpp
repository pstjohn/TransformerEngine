/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

void fp8_block_scaling_compute_partial_amax(Tensor tensor, Tensor amax, int64_t h, int64_t w,
                                            int64_t start_offset, int64_t block_len) {
  NVTE_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  NVTE_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  NVTE_CHECK(amax.scalar_type() == ScalarType::Float, "amax must be a float tensor");
  NVTE_CHECK(
      tensor.scalar_type() == ScalarType::Float || tensor.scalar_type() == ScalarType::BFloat16,
      "tensor must be a float or bfloat16 tensor");

  const auto tensor_cu = makeTransformerEngineTensor(tensor);
  auto amax_cu = makeTransformerEngineTensor(amax);

  // Compute strides from the 2D amax shape: contiguous layout -> stride(0) = sizes[1], stride(1) = 1
  auto amax_sizes = amax.sizes();
  int64_t amax_stride_0 = amax_sizes[1];
  int64_t amax_stride_1 = 1;

  nvte_fp8_block_scaling_compute_partial_amax(
      tensor_cu.data(), amax_cu.data(), static_cast<size_t>(h), static_cast<size_t>(w),
      amax_stride_0, amax_stride_1, static_cast<size_t>(start_offset),
      static_cast<size_t>(block_len), getCurrentCUDAStreamRaw(tensor.get_device_index()));
}

void fp8_block_scaling_partial_cast(Tensor inp, Tensor out, Tensor scale, int64_t h, int64_t w,
                                    int64_t start_offset, int64_t block_len, int64_t out_dtype) {
  NVTE_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  NVTE_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  NVTE_CHECK(scale.scalar_type() == ScalarType::Float, "scale must be a float tensor");
  NVTE_CHECK(inp.scalar_type() == ScalarType::Float || inp.scalar_type() == ScalarType::BFloat16,
             "input must be a float or bfloat16 tensor");
  NVTE_CHECK(out.scalar_type() == ScalarType::Byte, "output must be a uint8 tensor");
  auto te_out_dtype = static_cast<transformer_engine::DType>(out_dtype);
  NVTE_CHECK(te_out_dtype == transformer_engine::DType::kFloat8E4M3 ||
                 te_out_dtype == transformer_engine::DType::kFloat8E5M2,
             "out_dtype must be kFloat8E4M3 or kFloat8E5M2");

  const auto inp_cu = makeTransformerEngineTensor(inp);
  auto out_cu = makeTransformerEngineTensor(out);
  const auto scale_cu = makeTransformerEngineTensor(scale);

  auto scale_sizes = scale.sizes();
  int64_t scale_stride_0 = scale_sizes[1];
  int64_t scale_stride_1 = 1;

  nvte_fp8_block_scaling_partial_cast(
      inp_cu.data(), out_cu.data(), scale_cu.data(), static_cast<size_t>(h), static_cast<size_t>(w),
      scale_stride_0, scale_stride_1, static_cast<size_t>(start_offset),
      static_cast<size_t>(block_len), static_cast<NVTEDType>(te_out_dtype),
      getCurrentCUDAStreamRaw(inp.get_device_index()));
}

void mxfp8_scaling_compute_partial_amax(Tensor input, Tensor amax_rowwise, Tensor amax_colwise,
                                        int64_t rows, int64_t cols, int64_t start_offset) {
  const auto input_cu = makeTransformerEngineTensor(input);
  auto amax_rowwise_cu = makeTransformerEngineTensor(amax_rowwise);
  auto amax_colwise_cu = makeTransformerEngineTensor(amax_colwise);

  nvte_mxfp8_scaling_compute_partial_amax(input_cu.data(), amax_rowwise_cu.data(),
                                          amax_colwise_cu.data(), static_cast<int>(rows),
                                          static_cast<int>(cols), static_cast<size_t>(start_offset),
                                          getCurrentCUDAStreamRaw(input.get_device_index()));
}

void mxfp8_scaling_partial_cast(Tensor input, Tensor output_rowwise, Tensor output_colwise,
                                Tensor scale_inv_rowwise, Tensor scale_inv_colwise, int64_t rows,
                                int64_t cols, int64_t start_offset) {
  const auto input_cu = makeTransformerEngineTensor(input);
  auto output_rowwise_cu = makeTransformerEngineTensor(output_rowwise);
  auto output_colwise_cu = makeTransformerEngineTensor(output_colwise);
  const auto scale_inv_rowwise_cu = makeTransformerEngineTensor(scale_inv_rowwise);
  const auto scale_inv_colwise_cu = makeTransformerEngineTensor(scale_inv_colwise);

  nvte_mxfp8_scaling_partial_cast(input_cu.data(), output_rowwise_cu.data(),
                                  output_colwise_cu.data(), scale_inv_rowwise_cu.data(),
                                  scale_inv_colwise_cu.data(), static_cast<int>(rows),
                                  static_cast<int>(cols), static_cast<size_t>(start_offset),
                                  getCurrentCUDAStreamRaw(input.get_device_index()));
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_IMPL(transformer_engine, CUDA, m) {
  m.impl("fp8_block_scaling_compute_partial_amax",
         TORCH_BOX(&transformer_engine::pytorch::stable::fp8_block_scaling_compute_partial_amax));
  m.impl("fp8_block_scaling_partial_cast",
         TORCH_BOX(&transformer_engine::pytorch::stable::fp8_block_scaling_partial_cast));
  m.impl("mxfp8_scaling_compute_partial_amax",
         TORCH_BOX(&transformer_engine::pytorch::stable::mxfp8_scaling_compute_partial_amax));
  m.impl("mxfp8_scaling_partial_cast",
         TORCH_BOX(&transformer_engine::pytorch::stable::mxfp8_scaling_partial_cast));
}
