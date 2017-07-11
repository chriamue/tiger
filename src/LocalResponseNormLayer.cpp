/*
 * LocalResponseNormLayer.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: blake
 */

#include "LocalResponseNormLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
namespace tgr {
void LocalResponseNormLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	// @todo revise the parallelism strategy
	for (size_t sample = 0, sample_count = in_data[0]->size();
			sample < sample_count; ++sample) {
		Storage &in = (*in_data[0])[sample];
		Storage &out = (*out_data[0])[sample];

		if (region == norm_region::across_channels) {
			forward_across(in, out);
		} else {
			forward_within(in, out);
		}
	}
}
void LocalResponseNormLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	CNN_UNREFERENCED_PARAMETER(in_data);
	CNN_UNREFERENCED_PARAMETER(out_data);
	CNN_UNREFERENCED_PARAMETER(out_grad);
	CNN_UNREFERENCED_PARAMETER(in_grad);
	throw nn_error("not implemented");
}
void LocalResponseNormLayer::forward_across(const Storage &in, Storage &out) {
	vectorize::fill(&in_square[0], in_square.size(), float { 0 });

	for (int i = 0; i < size / 2; i++) {
		int idx = in_shape(0, 0, i);
		add_square_sum(&in[idx], in_shape.area(), &in_square[0]);
	}

	int head = size / 2;
	long tail = static_cast<long>(head) - static_cast<long>(size);
	int channels = in_shape.z;
	const int wxh = in_shape.area();
	const float alpha_div_size = alpha / size;

	for (int i = 0; i < channels; i++, head++, tail++) {
		if (head < channels)
			add_square_sum(&in[in_shape(0, 0, head)], wxh, &in_square[0]);

		if (tail >= 0)
			sub_square_sum(&in[in_shape(0, 0, tail)], wxh, &in_square[0]);

		float *dst = &out[in_shape(0, 0, i)];
		const float *src = &in[in_shape(0, 0, i)];
		for (int j = 0; j < wxh; j++)
			dst[j] = src[j]
					* std::pow(float(1) + alpha_div_size * in_square[j], -beta);
	}
}

void LocalResponseNormLayer::forward_within(const Storage &in, Storage &out) {
	CNN_UNREFERENCED_PARAMETER(in);
	CNN_UNREFERENCED_PARAMETER(out);
	throw nn_error("not implemented");
}

void LocalResponseNormLayer::add_square_sum(const float *src, int size,
		float *dst) {
	for (int i = 0; i < size; i++)
		dst[i] += src[i] * src[i];
}
void LocalResponseNormLayer::sub_square_sum(const float *src, int size,
		float *dst) {
	for (int i = 0; i < size; i++)
		dst[i] -= src[i] * src[i];
}
}

