/*
 * LinearLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */
#include "LinearLayer.h"
#include "tiny_dnn/tiny_dnn.h"
namespace tgr {
LinearLayer::LinearLayer(int dim, float scale, float bias) :
		NeuralLayer("Linear Transform", { ChannelType::data }, {
				ChannelType::data }), dim_(dim), scale_(scale), bias_(bias) {
}

std::vector<aly::dim3> LinearLayer::getInputDimensions() const {
	return {aly::dim3(dim_, 1, 1)};
}

std::vector<aly::dim3> LinearLayer::getOutputDimensions() const {
	return {aly::dim3(dim_, 1, 1)};
}

void LinearLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	const Tensor &in = *in_data[0];
	Tensor &out = *out_data[0];
	// do nothing
	CNN_UNREFERENCED_PARAMETER(out);
// @todo revise the parallelism strategy
	tiny_dnn::for_i(dim_, [&](size_t i) {
		for (int sample = 0,
				sample_count = static_cast<int>(in.size());
				sample < sample_count; ++sample)
		out[sample][i] = scale_ * in[sample][i] + bias_;
	});
}
void LinearLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	Tensor &prev_delta = *in_grad[0];
	Tensor &curr_delta = *out_grad[0];

	CNN_UNREFERENCED_PARAMETER(in_data);
// @todo revise parallelism strategy
	for (int sample = 0; sample < static_cast<int>(prev_delta.size());
			++sample) {
		tiny_dnn::for_i(dim_, [&](size_t i) {
			prev_delta[sample][i] = curr_delta[sample][i] * scale_;
		});
	}
}
}

