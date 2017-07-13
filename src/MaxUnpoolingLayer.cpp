/*
 * MaxUnpoolingLayer.cpp
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#include "MaxUnpoolingLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
namespace tgr {
int MaxUnpoolingLayer::getFanInSize() const {
	return 1;
}
int MaxUnpoolingLayer::getFanOutSize() const {
	return in2out[0].size();
}
MaxUnpoolingLayer::MaxUnpoolingLayer(int in_width, int in_height,
		int in_channels, int unpooling_size, int stride) :
		NeuralLayer("Max Unpooling", { ChannelType::data },
				{ ChannelType::data }), unpool_size(unpooling_size), stride(
				stride), in(in_width, in_height, in_channels), out(
				unpool_out_dim(in_width, unpooling_size, stride),
				unpool_out_dim(in_height, unpooling_size, stride), in_channels) {
	worker_storage.in2outmax.resize(out.size());
	init_connection();
}

void MaxUnpoolingLayer::connect_kernel(int unpooling_size, int inx, int iny,
		int c) {
	int dxmax = static_cast<int>(std::min(unpooling_size, inx * stride - out.x));
	int dymax = static_cast<int>(std::min(unpooling_size, iny * stride - out.y));

	for (int dy = 0; dy < dymax; dy++) {
		for (int dx = 0; dx < dxmax; dx++) {
			int out_index = out(static_cast<int>(inx * stride + dx),
					static_cast<int>(iny * stride + dy), c);
			int in_index = in(inx, iny, c);

			if (in_index >= in2out.size())
				throw nn_error("index overflow");
			if (out_index >= out2in.size())
				throw nn_error("index overflow");
			out2in[out_index] = in_index;
			in2out[in_index].push_back(out_index);
		}
	}
}

void MaxUnpoolingLayer::init_connection() {
	in2out.resize(in.size());
	out2in.resize(out.size());

	worker_storage.in2outmax.resize(in.size());

	for (int c = 0; c < in.z; ++c)
		for (int y = 0; y < in.y; ++y)
			for (int x = 0; x < in.x; ++x)
				connect_kernel(static_cast<int>(unpool_size), x, y, c);
}

void MaxUnpoolingLayer::forwardPropagation(const std::vector<Tensor *> &in_data,std::vector<Tensor *> &out_data) {
	const Tensor &in = *in_data[0];
	Tensor & out = *out_data[0];
	for (size_t sample = 0; sample < in_data[0]->size(); sample++) {
		const Storage &in_vec = in[sample];
		Storage &out_vec = out[sample];
		std::vector<int> &max_idx = worker_storage.in2outmax;
		for_i(in2out.size(),
				[&](size_t i) {
					const auto &in_index = out2in[i];
					out_vec[i] = (max_idx[in_index] == i) ? in_vec[in_index] : float_t {0};
				});
	}
}

void MaxUnpoolingLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data,
		std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	Tensor &prev_delta = *in_grad[0];
	Tensor &curr_delta = *out_grad[0];
	for (int sample = 0; sample < in_data[0]->size(); sample++) {
		Storage &prev_delta_vec = prev_delta[sample];
		Storage &curr_delta_vec = curr_delta[sample];

		std::vector<int> &max_idx = worker_storage.in2outmax;

		for_(parallelize, 0, in2out.size(),
				[&](const blocked_range &r) {
					for (size_t i = r.begin(); i != r.end(); i++) {
						int outi = out2in[i];
						prev_delta_vec[i] =
						(max_idx[outi] == i) ? curr_delta_vec[outi] : float_t {0};
					}
				});
	}
}
}

