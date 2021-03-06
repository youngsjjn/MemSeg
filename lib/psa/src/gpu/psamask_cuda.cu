#include <torch/serialize/tensor.h>

// CUDA: grid stride looping
#ifndef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#endif

__global__ void psamask_collect_forward_cuda(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const float* mask_data, float* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}

__global__ void psamask_distribute_forward_cuda(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const float* mask_data, float* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}

__global__ void psamask_collect_backward_cuda(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const float* buffer_diff, float* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
      	mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w];
      }
    }
  }
}

__global__ void psamask_distribute_backward_cuda(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const float* buffer_diff, float* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
      	mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)];
      }
    }
  }
}

void psamask_forward_cuda(const int psa_type, const at::Tensor& input, at::Tensor& output, const int num_, const int feature_H_, const int feature_W_, const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_)
{
    int nthreads = num_ * feature_H_ * feature_W_;
    const float* input_data = input.data<float>();
    float* output_data = output.data<float>();
    if(psa_type == 0)
        psamask_collect_forward_cuda<<<nthreads, 512>>>(nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_, input_data, output_data);
    else
        psamask_distribute_forward_cuda<<<nthreads, 512>>>(nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_, input_data, output_data);
}

void psamask_backward_cuda(const int psa_type, const at::Tensor& grad_output, at::Tensor& grad_input, const int num_, const int feature_H_, const int feature_W_, const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_)
{
    int nthreads = num_ * feature_H_ * feature_W_;
    const float* grad_output_data = grad_output.data<float>();
    float* grad_input_data = grad_input.data<float>();
    if(psa_type == 0)
        psamask_collect_backward_cuda<<<nthreads, 512>>>(nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_, grad_output_data, grad_input_data);
    else
        psamask_distribute_backward_cuda<<<nthreads, 512>>>(nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_, grad_output_data, grad_input_data);
}
