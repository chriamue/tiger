/*
 * AveragePoolingLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "AveragePoolingLayer.h"
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/layers/layer.h"
using namespace tiny_dnn;
using namespace aly;
namespace tgr {

// forward_propagation
inline void tiny_average_pooling_kernel(bool parallelize,
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data,
		const dim3& out_dim, float_t scale_factor,
		std::vector<typename PartialConnectedLayer::wi_connections> &out2wi) {
	tiny_dnn::for_i(parallelize, in_data[0]->size(),
			[&](size_t sample) {
				const Storage &in = (*in_data[0])[sample];
				const Storage &W = (*in_data[1])[0];
				const Storage &b = (*in_data[2])[0];
				Storage &out = (*out_data[0])[sample];
				auto oarea = out_dim.area();
				size_t idx = 0;
				for (int d = 0; d < out_dim.z; ++d) {
					float_t weight = W[d] * scale_factor;
					float_t bias = b[d];
					for (int i = 0; i < oarea; ++i, ++idx) {
						const auto &connections = out2wi[idx];
						float_t value {0};
						for (auto connection : connections) value += in[connection.second];
						value *= weight;
						value += bias;
						out[idx] = value;
					}
				}
				assert(out.size() == out2wi.size());
			});
}
void AveragePoolingLayer::getStencilInput(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	wo_connections outarray = out2wi_[out_(pos)];
	stencil.resize(outarray.size());
	for (int i = 0; i < outarray.size(); i++) {
		stencil[i] = in_(outarray[i].second);
	}
}
void AveragePoolingLayer::getStencilWeight(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	wo_connections outarray = out2wi_[out_(pos)];
	stencil.resize(outarray.size());
	for (int i = 0; i < outarray.size(); i++) {
		stencil[i] = in_(outarray[i].first);
	}
}
bool AveragePoolingLayer::getStencilBias(const aly::int3& pos,
		aly::int3& stencil) const {
	if (out2bias_.size() > 0) {
		stencil = in_(out2bias_(out_(pos)));
		return true;
	} else {
		return false;
	}
}

// back_propagation
inline void tiny_average_pooling_back_kernel(bool parallelize,
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad, const dim3 &in_dim,
		float_t scale_factor,
		std::vector<typename PartialConnectedLayer::io_connections> &weight2io,
		std::vector<typename PartialConnectedLayer::wo_connections> &in2wo,
		std::vector<std::vector<int>> &bias2out) {
	CNN_UNREFERENCED_PARAMETER(out_data);
	tiny_dnn::for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
		const Storage &prev_out = (*in_data[0])[sample];
		const Storage &W = (*in_data[1])[0];
		Storage &dW = (*in_grad[1])[sample];
		Storage &db = (*in_grad[2])[sample];
		Storage &prev_delta = (*in_grad[0])[sample];
		Storage &curr_delta = (*out_grad[0])[sample];

		auto inarea = in_dim.area();
		size_t idx = 0;
		for (size_t i = 0; i < in_dim.z; ++i) {
			float_t weight = W[i] * scale_factor;
			for (size_t j = 0; j < inarea; ++j, ++idx) {
				prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
			}
		}

		for (size_t i = 0; i < weight2io.size(); ++i) {
			const auto &connections = weight2io[i];
			float_t diff {0};

			for (auto connection : connections)
			diff += prev_out[connection.first] * curr_delta[connection.second];

			dW[i] += diff * scale_factor;
		}

		for (size_t i = 0; i < bias2out.size(); i++) {
			const std::vector<int> &outs = bias2out[i];
			float_t diff {0};

			for (auto o : outs) diff += curr_delta[o];

			db[i] += diff;
		}
	});
}
std::vector<aly::dim3> AveragePoolingLayer::getInputDimensions() const {
	return {in_, w_, dim3(1, 1, out_.z)};
}

std::vector<aly::dim3> AveragePoolingLayer::getOutputDimensions() const {
	return {out_};
}

void AveragePoolingLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	tiny_average_pooling_kernel(parallelize, in_data, out_data, out_,
			Base::scale_factor_, Base::out2wi_);
}

void AveragePoolingLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	tiny_average_pooling_back_kernel(parallelize, in_data, out_data, out_grad,
			in_grad, in_, Base::scale_factor_, Base::weight2io_, Base::in2wo_,
			Base::bias2out_);
}

std::pair<int, int> AveragePoolingLayer::pool_size() const {
	return std::make_pair(pool_size_x_, pool_size_y_);
}
int AveragePoolingLayer::pool_out_dim(int in_size, int pooling_size,
		int stride) {
	return static_cast<int>(std::ceil(
			(static_cast<float_t>(in_size) - pooling_size) / stride) + 1);
}

void AveragePoolingLayer::init_connection(int pooling_size_x,
		int pooling_size_y) {
	for (int c = 0; c < in_.z; ++c) {
		for (int y = 0; y < in_.y - pooling_size_y + 1; y += stride_y_) {
			for (int x = 0; x < in_.x - pooling_size_x + 1; x += stride_x_) {
				connect_kernel(pooling_size_x, pooling_size_y, x, y, c);
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

void AveragePoolingLayer::connect_kernel(int pooling_size_x, int pooling_size_y,
		int x, int y, int inc) {
	int dymax = std::min(pooling_size_y, (int) (in_.y - y));
	int dxmax = std::min(pooling_size_x, (int) (in_.x - x));
	int dstx = x / stride_x_;
	int dsty = y / stride_y_;
	int outidx = out_(dstx, dsty, inc);
	for (int dy = 0; dy < dymax; ++dy) {
		for (int dx = 0; dx < dxmax; ++dx) {
			this->connect_weight(in_(x + dx, y + dy, inc), outidx, inc);
		}
	}
}
AveragePoolingLayer::AveragePoolingLayer(int in_width, int in_height,
		int in_channels, int pool_size_x, int pool_size_y, int stride_x,
		int stride_y, Padding pad_type) :
		PartialConnectedLayer("Average Pool",
				in_width * in_height * in_channels,
				conv_out_length(in_width, pool_size_x, stride_x,
						static_cast<padding>(pad_type))
						* conv_out_length(in_height, pool_size_y, stride_y,
								static_cast<padding>(pad_type)) * in_channels,
				in_channels, in_channels,
				float_t(1) / (pool_size_x * pool_size_y)), stride_x_(stride_x), stride_y_(
				stride_y), pool_size_x_(pool_size_x), pool_size_y_(pool_size_y), pad_type_(
				pad_type), in_(in_width, in_height, in_channels), out_(
				conv_out_length(in_width, pool_size_x, stride_x,
						static_cast<padding>(pad_type)),
				conv_out_length(in_height, pool_size_y, stride_y,
						static_cast<padding>(pad_type)), in_channels), w_(
				pool_size_x, pool_size_y, in_channels) {
	if ((in_width % pool_size_x) || (in_height % pool_size_y)) {
		pooling_size_mismatch(in_width, in_height, pool_size_x, pool_size_y);
	}

	init_connection(pool_size_x, pool_size_y);
}

}

