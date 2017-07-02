/*
 * MaxPoolingLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */
#include "MaxPoolingLayer.h"
using namespace aly;
using namespace tiny_dnn;
using namespace tiny_dnn::core;
namespace tgr {
MaxPoolingLayer::MaxPoolingLayer(int in_width, int in_height, int in_channels,
		int pooling_size_x, int pooling_size_y, int stride_x, int stride_y,
		Padding pad_type, BackendType backend_type) :
		NeuralLayer("Max Pooling", { ChannelType::data }, { ChannelType::data }) {
	set_maxpool_params(shape3d(in_width, in_height, in_channels),
			shape3d(
					conv_out_length(in_width, pooling_size_x, stride_x,
							static_cast<padding>(pad_type)),
					conv_out_length(in_height, pooling_size_y, stride_y,
							static_cast<padding>(pad_type)), in_channels),
			pooling_size_x, pooling_size_y, stride_x, stride_y,
			static_cast<padding>(pad_type));

	init_connection();
	init_backend(backend_type);
	NeuralLayer::setBackendType(backend_type);
}

int MaxPoolingLayer::getFanInSize() const {
	return static_cast<int>(params_.out2in[0].size());
}

int MaxPoolingLayer::getFanOutSize() const {
	return 1;
}

void MaxPoolingLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	// forward convolutional op context
	fwd_ctx_.set_in_out(in_data, out_data);
	fwd_ctx_.setParallelize(parallelize);
	fwd_ctx_.setEngine(static_cast<backend_t>(getBackendType()));

	// launch convolutional kernel
	kernel_fwd_->compute(fwd_ctx_);
}

void MaxPoolingLayer::backwardPropagation(
		const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data,
		std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	// backward convolutional op context
	bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
	bwd_ctx_.setParallelize(parallelize);
	bwd_ctx_.setEngine(static_cast<backend_t>(getBackendType()));

	// launch convolutional kernel
	kernel_back_->compute(bwd_ctx_);
}

std::vector<dim3> MaxPoolingLayer::getInputDimensions() const {
	return {Convert(params_.in)};
}

std::vector<dim3> MaxPoolingLayer::getOutputDimensions() const {
	return {Convert(params_.out)};
}

std::pair<int, int> MaxPoolingLayer::pool_size() const {
	return std::make_pair(params_.pool_size_x, params_.pool_size_y);
}

void MaxPoolingLayer::setSampleCount(size_t sample_count) {
	NeuralLayer::setSampleCount(sample_count);
	params_.out2inmax.resize(sample_count,std::vector<uint32_t>(params_.out.size()));
}
void MaxPoolingLayer::connect_kernel(int pooling_size_x, int pooling_size_y,
		int outx, int outy, int c) {
	int dxmax = static_cast<int>(std::min(static_cast<int>(pooling_size_x),
			(int) (params_.in.width_ - outx * params_.stride_x)));

	int dymax = static_cast<int>(std::min(static_cast<int>(pooling_size_y),
			(int) (params_.in.height_ - outy * params_.stride_y)));

	for (int dy = 0; dy < dymax; dy++) {
		for (int dx = 0; dx < dxmax; dx++) {
			int in_index = params_.in.get_index(
					static_cast<int>(outx * params_.stride_x + dx),
					static_cast<int>(outy * params_.stride_y + dy), c);
			int out_index = params_.out.get_index(outx, outy, c);

			if (in_index >= params_.in2out.size()) {
				throw nn_error("index overflow");
			}
			if (out_index >= params_.out2in.size()) {
				throw nn_error("index overflow");
			}
			params_.in2out[in_index] = out_index;
			params_.out2in[out_index].push_back(in_index);
		}
	}
}

void MaxPoolingLayer::init_connection() {
	params_.in2out.resize(params_.in.size());
	params_.out2in.resize(params_.out.size());

	for (int c = 0; c < params_.in.depth_; ++c) {
		for (int y = 0; y < params_.out.height_; ++y) {
			for (int x = 0; x < params_.out.width_; ++x) {
				connect_kernel(params_.pool_size_x, params_.pool_size_y, x, y,
						c);
			}
		}
	}
}

void MaxPoolingLayer::init_backend(BackendType backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params_);
	if (static_cast<backend_t>(backend_type) == backend_t::internal
			|| static_cast<backend_t>(backend_type) == backend_t::nnpack
			|| static_cast<backend_t>(backend_type) == backend_t::avx) {
		kernel_fwd_.reset(new MaxPoolOp(ctx));
		kernel_back_.reset(new MaxPoolGradOp(ctx));
		return;
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}

void MaxPoolingLayer::set_maxpool_params(const shape3d &in, const shape3d &out,
		int pooling_size_x, int pooling_size_y, int stride_x, int stride_y,
		padding pad_type) {
	params_.in = in;
	params_.out = out;
	params_.pool_size_x = pooling_size_x;
	params_.pool_size_y = pooling_size_y;
	params_.stride_x = stride_x;
	params_.stride_y = stride_y;
	params_.pad_type = pad_type;
}
}

