/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/fused_rope.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor fused_rope_forward(Tensor input, Tensor freqs, std::optional<Tensor> start_positions,
                          int64_t qkv_format, bool interleaved, std::optional<Tensor> cu_seqlens,
                          int64_t cp_size, int64_t cp_rank) {
  NVTE_CHECK(freqs.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(freqs.sizes()[1] == 1 && freqs.sizes()[2] == 1,
             "expected the second and third dims of the freqs tensor equal 1");
  NVTE_CHECK(freqs.scalar_type() == ScalarType::Float, "Dtype of the freqs tensor must be float");

  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);

  // output
  auto input_sizes = input.sizes();
  std::vector<int64_t> out_shape(input_sizes.begin(), input_sizes.end());
  auto output = allocateStableTensor(out_shape, input.scalar_type(), input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto output_cu = makeTransformerEngineTensor(output);

  auto start_positions_cu = transformer_engine::TensorWrapper();
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(*start_positions);
    NVTE_CHECK(start_positions_cu.ndim() == 1, "expected 1D tensor");
  }

  if (nvte_qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(input.dim() == 3, "expected 3D tensor");
    NVTE_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");
    NVTE_CHECK(cu_seqlens->dim() == 1, "expected 1D tensor");
    NVTE_CHECK(input.sizes()[2] >= freqs.sizes()[3],
               "expected the last dim of the input tensor equals or is "
               "greater than the freqs tensor");

    const int h = input.sizes()[1];
    const int d = input.sizes()[2];
    const int stride_t = input.stride(0);
    const int stride_h = input.stride(1);
    const int stride_d = input.stride(2);
    const int b = cu_seqlens->sizes()[0] - 1;
    const int max_s = freqs.sizes()[0];
    const int d2 = freqs.sizes()[3];

    auto cu_seqlens_cu = makeTransformerEngineTensor(*cu_seqlens);

    nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                            start_positions_cu.data(), output_cu.data(), nvte_qkv_format,
                            interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank),
                            max_s, b, h, d, d2, stride_t, /*stride_b=*/0, stride_h, stride_d,
                            getCurrentCUDAStreamRaw(input.get_device_index()));

    return output;
  }

  NVTE_CHECK(input.dim() == 4, "expected 4D tensor");
  const int s = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.sizes()[0] : input.sizes()[1];
  const int b = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.sizes()[1] : input.sizes()[0];
  const int h = input.sizes()[2];
  const int d = input.sizes()[3];
  const int stride_s =
      nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.stride(0) : input.stride(1);
  const int stride_b =
      nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.stride(1) : input.stride(0);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  const int d2 = freqs.sizes()[3];

  NVTE_CHECK(s * static_cast<int>(cp_size) <= freqs.sizes()[0],
             "expected freqs tensor has a longer sequence length than input");
  NVTE_CHECK(d >= d2,
             "expected the last dim of the input tensor equals or is "
             "greater than the freqs tensor");

  auto cu_seqlens_cu = transformer_engine::TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                          start_positions_cu.data(), output_cu.data(), nvte_qkv_format, interleaved,
                          static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
                          stride_s, stride_b, stride_h, stride_d,
                          getCurrentCUDAStreamRaw(input.get_device_index()));

  return output;
}

Tensor fused_rope_backward(Tensor output_grads, Tensor freqs, std::optional<Tensor> start_positions,
                           int64_t qkv_format, bool interleaved, std::optional<Tensor> cu_seqlens,
                           int64_t cp_size, int64_t cp_rank) {
  NVTE_CHECK(freqs.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(freqs.sizes()[1] == 1 && freqs.sizes()[2] == 1,
             "expected the second and third dims of the freqs tensor equal 1");
  NVTE_CHECK(freqs.scalar_type() == ScalarType::Float, "Dtype of the freqs tensor must be float");

  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);

  auto og_sizes = output_grads.sizes();
  std::vector<int64_t> out_shape(og_sizes.begin(), og_sizes.end());
  auto input_grads =
      allocateStableTensor(out_shape, output_grads.scalar_type(), output_grads.get_device_index());

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto input_grads_cu = makeTransformerEngineTensor(input_grads);

  auto start_positions_cu = transformer_engine::TensorWrapper();
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(*start_positions);
    NVTE_CHECK(start_positions_cu.ndim() == 1, "expected 1D tensor");
  }

  if (nvte_qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(output_grads.dim() == 3, "expected 3D tensor");
    NVTE_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");
    NVTE_CHECK(cu_seqlens->dim() == 1, "expected 1D tensor");
    NVTE_CHECK(output_grads.sizes()[2] >= freqs.sizes()[3],
               "expected the last dim of the output_grads tensor equals or is "
               "greater than the freqs tensor");

    const int h = output_grads.sizes()[1];
    const int d = output_grads.sizes()[2];
    const int stride_t = output_grads.stride(0);
    const int stride_h = output_grads.stride(1);
    const int stride_d = output_grads.stride(2);
    const int b = cu_seqlens->sizes()[0] - 1;
    const int max_s = freqs.sizes()[0];
    const int d2 = freqs.sizes()[3];

    auto cu_seqlens_cu = makeTransformerEngineTensor(*cu_seqlens);

    nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                             start_positions_cu.data(), input_grads_cu.data(), nvte_qkv_format,
                             interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank),
                             max_s, b, h, d, d2, stride_t,
                             /*stride_b=*/0, stride_h, stride_d,
                             getCurrentCUDAStreamRaw(output_grads.get_device_index()));

    return input_grads;
  }

  NVTE_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  const int s = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.sizes()[0]
                                                              : output_grads.sizes()[1];
  const int b = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.sizes()[1]
                                                              : output_grads.sizes()[0];
  const int h = output_grads.sizes()[2];
  const int d = output_grads.sizes()[3];
  const int stride_s = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.stride(0)
                                                                     : output_grads.stride(1);
  const int stride_b = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.stride(1)
                                                                     : output_grads.stride(0);
  const int stride_h = output_grads.stride(2);
  const int stride_d = output_grads.stride(3);
  const int d2 = freqs.sizes()[3];

  NVTE_CHECK(s * static_cast<int>(cp_size) <= freqs.sizes()[0],
             "expected freqs tensor has a longer sequence length than output_grads");
  NVTE_CHECK(d >= d2,
             "expected the last dim of the output_grads tensor equals or is "
             "greater than the freqs tensor");

  auto cu_seqlens_cu = transformer_engine::TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                           start_positions_cu.data(), input_grads_cu.data(), nvte_qkv_format,
                           interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b,
                           h, d, d2, stride_s, stride_b, stride_h, stride_d,
                           getCurrentCUDAStreamRaw(output_grads.get_device_index()));

  return input_grads;
}

std::tuple<Tensor, Tensor, Tensor> fused_qkv_rope_forward(Tensor qkv_input, Tensor q_freqs,
                                                          Tensor k_freqs,
                                                          std::optional<Tensor> start_positions,
                                                          std::vector<int64_t> qkv_split_arg_list,
                                                          int64_t qkv_format, bool interleaved,
                                                          int64_t cp_size, int64_t cp_rank) {
  NVTE_CHECK(q_freqs.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(q_freqs.sizes()[1] == 1 && q_freqs.sizes()[2] == 1,
             "expected the second and third dims of the freqs tensor equal 1");
  NVTE_CHECK(q_freqs.scalar_type() == ScalarType::Float, "Dtype of the freqs tensor must be float");
  NVTE_CHECK(k_freqs.dim() == 4, "expected 4D tensor");
  NVTE_CHECK(k_freqs.sizes()[1] == 1 && k_freqs.sizes()[2] == 1,
             "expected the second and third dims of the freqs tensor equal 1");
  NVTE_CHECK(k_freqs.scalar_type() == ScalarType::Float, "Dtype of the freqs tensor must be float");

  NVTE_CHECK(qkv_split_arg_list.size() >= 3, "qkv_split_arg_list must have at least 3 elements");

  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);
  auto qkv_sizes = qkv_input.sizes();
  int32_t dev_idx = qkv_input.get_device_index();

  // output
  std::vector<int64_t> q_out_size(qkv_sizes.begin(), qkv_sizes.end());
  q_out_size[2] = q_out_size[2] * qkv_split_arg_list[0] / qkv_split_arg_list[1];
  q_out_size[3] = qkv_split_arg_list[1];
  auto q_out = allocateStableTensor(q_out_size, qkv_input.scalar_type(), dev_idx);

  std::vector<int64_t> k_out_size(qkv_sizes.begin(), qkv_sizes.end());
  k_out_size[3] = qkv_split_arg_list[1];
  auto k_out = allocateStableTensor(k_out_size, qkv_input.scalar_type(), dev_idx);

  std::vector<int64_t> v_out_size(qkv_sizes.begin(), qkv_sizes.end());
  v_out_size[3] = qkv_split_arg_list[2];
  auto v_out = allocateStableTensor(v_out_size, qkv_input.scalar_type(), dev_idx);

  auto qkv_cu = makeTransformerEngineTensor(qkv_input);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto q_out_cu = makeTransformerEngineTensor(q_out);
  auto k_out_cu = makeTransformerEngineTensor(k_out);
  auto v_out_cu = makeTransformerEngineTensor(v_out);

  auto start_positions_cu = transformer_engine::TensorWrapper();
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(*start_positions);
  }

  NVTE_CHECK(qkv_input.dim() == 4, "expected 4D input tensor");

  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? qkv_input.sizes()[0] : qkv_input.sizes()[1];
  const int b = is_sbhd ? qkv_input.sizes()[1] : qkv_input.sizes()[0];
  const int h = qkv_input.sizes()[2];
  const int d = static_cast<int>(qkv_split_arg_list[2]);
  const int d2 = q_freqs.sizes()[3];

  nvte_fused_qkv_rope_forward(
      qkv_cu.data(), q_freqs_cu.data(), k_freqs_cu.data(), start_positions_cu.data(),
      q_out_cu.data(), k_out_cu.data(), v_out_cu.data(), nvte_qkv_format, interleaved,
      static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
      static_cast<int>(qkv_split_arg_list[0]), static_cast<int>(qkv_split_arg_list[1]),
      static_cast<int>(qkv_split_arg_list[2]), getCurrentCUDAStreamRaw(dev_idx));

  return std::make_tuple(q_out, k_out, v_out);
}

Tensor fused_qkv_rope_backward(Tensor q_grad_out, Tensor k_grad_out, Tensor v_grad_out,
                               Tensor q_freqs, Tensor k_freqs,
                               std::vector<int64_t> qkv_split_arg_list, int64_t qkv_format,
                               bool interleaved, int64_t cp_size, int64_t cp_rank) {
  NVTE_CHECK(qkv_split_arg_list.size() >= 3, "qkv_split_arg_list must have at least 3 elements");

  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);
  int32_t dev_idx = q_grad_out.get_device_index();
  auto q_sizes = q_grad_out.sizes();

  auto total_hd = (q_grad_out.sizes()[2] + k_grad_out.sizes()[2] + v_grad_out.sizes()[2]) *
                  q_grad_out.sizes()[3];
  auto total_d = qkv_split_arg_list[0] + qkv_split_arg_list[1] + qkv_split_arg_list[2];
  std::vector<int64_t> qkv_grad_size(q_sizes.begin(), q_sizes.end());
  qkv_grad_size[2] = total_hd / total_d;
  qkv_grad_size[3] = total_d;
  auto qkv_grad_input = allocateStableTensor(qkv_grad_size, q_grad_out.scalar_type(), dev_idx);

  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? q_grad_out.sizes()[0] : q_grad_out.sizes()[1];
  const int b = is_sbhd ? q_grad_out.sizes()[1] : q_grad_out.sizes()[0];
  const int h = qkv_grad_size[2];
  const int d = static_cast<int>(qkv_split_arg_list[2]);
  const int d2 = q_freqs.sizes()[3];

  auto q_grad_out_cu = makeTransformerEngineTensor(q_grad_out);
  auto k_grad_out_cu = makeTransformerEngineTensor(k_grad_out);
  auto v_grad_out_cu = makeTransformerEngineTensor(v_grad_out);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto qkv_grad_cu = makeTransformerEngineTensor(qkv_grad_input);

  nvte_fused_qkv_rope_backward(
      q_grad_out_cu.data(), k_grad_out_cu.data(), v_grad_out_cu.data(), q_freqs_cu.data(),
      k_freqs_cu.data(), qkv_grad_cu.data(), nvte_qkv_format, interleaved,
      static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
      static_cast<int>(qkv_split_arg_list[0]), static_cast<int>(qkv_split_arg_list[1]),
      static_cast<int>(qkv_split_arg_list[2]), getCurrentCUDAStreamRaw(dev_idx));

  return qkv_grad_input;
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_IMPL(transformer_engine, CUDA, m) {
  m.impl("fused_rope_forward", TORCH_BOX(&transformer_engine::pytorch::stable::fused_rope_forward));
  m.impl("fused_rope_backward",
         TORCH_BOX(&transformer_engine::pytorch::stable::fused_rope_backward));
  m.impl("fused_qkv_rope_forward",
         TORCH_BOX(&transformer_engine::pytorch::stable::fused_qkv_rope_forward));
  m.impl("fused_qkv_rope_backward",
         TORCH_BOX(&transformer_engine::pytorch::stable::fused_qkv_rope_backward));
}
