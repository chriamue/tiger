/*
 * AverageUnpoolingLayer.cpp
 *
 *  Created on: Jun 26, 2017
 *      Author: blake
 */

#include "AverageUnpoolingLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
namespace tgr {
// forward_propagation
void tiny_average_unpooling_kernel(bool parallelize,
		const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data,
		const aly::dim3 &out_dim,
		float scale_factor,
		std::vector<typename PartialConnectedLayer::wi_connections> &out2wi) {
	CNN_UNREFERENCED_PARAMETER(scale_factor);
	for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
		const Storage &in = (*in_data[0])[sample];
		const Storage &W = (*in_data[1])[0];
		const Storage &b = (*in_data[2])[0];
		Storage &out = (*out_data[0])[sample];
		auto oarea = out_dim.x*out_dim.y;
		size_t idx = 0;
		for (size_t d = 0; d < out_dim.z; ++d) {
			float weight = W[d];  // * scale_factor;
			float bias = b[d];
			for (size_t i = 0; i < oarea; ++i, ++idx) {
				const auto &connections = out2wi[idx];
				float value {0};
				for (auto connection : connections) value += in[connection.second];
				value *= weight;
				value += bias;
				out[idx] = value;
			}
		}

		assert(out.size() == out2wi.size());
	});
}

// back_propagation
void tiny_average_unpooling_back_kernel(bool parallelize,
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad, const aly::dim3 &in_dim,
		float scale_factor,
		std::vector<typename PartialConnectedLayer::io_connections> &weight2io,
		std::vector<typename PartialConnectedLayer::wo_connections> &in2wo,
		std::vector<std::vector<int>> &bias2out) {
	CNN_UNREFERENCED_PARAMETER(out_data);
	CNN_UNREFERENCED_PARAMETER(scale_factor);
	for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
		const Storage &prev_out = (*in_data[0])[sample];
		const Storage &W = (*in_data[1])[0];
		Storage &dW = (*in_grad[1])[sample];
		Storage &db = (*in_grad[2])[sample];
		Storage &prev_delta = (*in_grad[0])[sample];
		Storage &curr_delta = (*out_grad[0])[sample];

		auto inarea = in_dim.x*in_dim.y;
		size_t idx = 0;
		for (size_t i = 0; i < in_dim.z; ++i) {
			float weight = W[i];  // * scale_factor;
			for (size_t j = 0; j < inarea; ++j, ++idx) {
				prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
			}
		}

		for (size_t i = 0; i < weight2io.size(); ++i) {
			const auto &connections = weight2io[i];
			float diff {0};

			for (auto connection : connections)
			diff += prev_out[connection.first] * curr_delta[connection.second];

			dW[i] += diff;  // * scale_factor;
		}

		for (size_t i = 0; i < bias2out.size(); i++) {
			const std::vector<int> &outs = bias2out[i];
			float diff {0};

			for (auto o : outs) diff += curr_delta[o];

			db[i] += diff;
		}
	});
}

AverageUnpoolingLayer::AverageUnpoolingLayer(int in_width, int in_height,
		int in_channels, int pooling_size) :
		PartialConnectedLayer("Average Un-pooling",
				in_width * in_height * in_channels,
				in_width * in_height * in_channels * sqr(pooling_size),
				in_channels, in_channels, float(1) * sqr(pooling_size)), stride_(
				pooling_size), in_(in_width, in_height, in_channels), out_(
				in_width * pooling_size, in_height * pooling_size, in_channels), w_(
				pooling_size, (in_height == 1 ? 1 : pooling_size), in_channels) {
	init_connection(pooling_size);
}
std::vector<aly::dim3> AverageUnpoolingLayer::getInputDimensions() const {
	return {in_, w_,aly::dim3(1, 1, out_.z)};
}

std::vector<aly::dim3> AverageUnpoolingLayer::getOutputDimensions() const {
	return {out_};
}
void AverageUnpoolingLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	tiny_average_unpooling_kernel(parallelize, in_data, out_data, out_,
			PartialConnectedLayer::scale_factor_,
			PartialConnectedLayer::out2wi_);
}

void AverageUnpoolingLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	tiny_average_unpooling_back_kernel(parallelize, in_data, out_data, out_grad,
			in_grad, in_, PartialConnectedLayer::scale_factor_,
			PartialConnectedLayer::weight2io_, PartialConnectedLayer::in2wo_,
			PartialConnectedLayer::bias2out_);
}

int AverageUnpoolingLayer::unpool_out_dim(int in_size, int pooling_size,
		int stride) {
	return static_cast<int>((in_size - 1) * stride + pooling_size);
}

void AverageUnpoolingLayer::init_connection(int pooling_size) {
	for (int c = 0; c < in_.z; ++c) {
		for (int y = 0; y < in_.y; ++y) {
			for (int x = 0; x < in_.x; ++x) {
				connect_kernel(pooling_size, x, y, c);
			}
		}
	}

	for (int c = 0; c < in_.z; ++c) {
		for (int y = 0; y < out_.y; ++y) {
			for (int x = 0; x < out_.x; ++x) {
				this->connect_bias(c, out_(x, y, c));
			}
		}
	}
}

void AverageUnpoolingLayer::connect_kernel(int pooling_size, int x, int y,
		int inc) {
	int dymax = std::min(pooling_size, out_.y - y);
	int dxmax = std::min(pooling_size, out_.x - x);
	int dstx = x * stride_;
	int dsty = y * stride_;
	int inidx = in_(x, y, inc);
	for (int dy = 0; dy < dymax; ++dy) {
		for (int dx = 0; dx < dxmax; ++dx) {
			this->connect_weight(inidx, out_(dstx + dx, dsty + dy, inc), inc);
		}
	}
}

}

