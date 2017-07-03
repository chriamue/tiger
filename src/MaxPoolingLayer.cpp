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
	return static_cast<int>(params.out2in[0].size());
}

int MaxPoolingLayer::getFanOutSize() const {
	return 1;
}

void MaxPoolingLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	// forward convolutional op context
	fwd_ctx.set_in_out(in_data, out_data);
	fwd_ctx.setParallelize(parallelize);
	fwd_ctx.setEngine(static_cast<backend_t>(getBackendType()));

	// launch convolutional kernel
	kernel_fwd->compute(fwd_ctx);
}

void MaxPoolingLayer::backwardPropagation(
		const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data,
		std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	// backward convolutional op context
	bwd_ctx.set_in_out(in_data, out_data, out_grad, in_grad);
	bwd_ctx.setParallelize(parallelize);
	bwd_ctx.setEngine(static_cast<backend_t>(getBackendType()));

	// launch convolutional kernel
	kernel_back->compute(bwd_ctx);
}

std::vector<dim3> MaxPoolingLayer::getInputDimensions() const {
	return {Convert(params.in)};
}

std::vector<dim3> MaxPoolingLayer::getOutputDimensions() const {
	return {Convert(params.out)};
}

std::pair<int, int> MaxPoolingLayer::pool_size() const {
	return std::make_pair(params.pool_size_x, params.pool_size_y);
}

void MaxPoolingLayer::setSampleCount(size_t sample_count) {
	NeuralLayer::setSampleCount(sample_count);
	params.out2inmax.resize(sample_count,std::vector<uint32_t>(params.out.size()));
}
void MaxPoolingLayer::connect_kernel(int pooling_size_x, int pooling_size_y,
		int outx, int outy, int c) {
	int dxmax = static_cast<int>(std::min(static_cast<int>(pooling_size_x),
			(int) (params.in.width - outx * params.stride_x)));

	int dymax = static_cast<int>(std::min(static_cast<int>(pooling_size_y),
			(int) (params.in.height - outy * params.stride_y)));

	for (int dy = 0; dy < dymax; dy++) {
		for (int dx = 0; dx < dxmax; dx++) {
			int in_index = params.in.get_index(
					static_cast<int>(outx * params.stride_x + dx),
					static_cast<int>(outy * params.stride_y + dy), c);
			int out_index = params.out.get_index(outx, outy, c);

			if (in_index >= params.in2out.size()) {
				throw nn_error("index overflow");
			}
			if (out_index >= params.out2in.size()) {
				throw nn_error("index overflow");
			}
			params.in2out[in_index] = out_index;
			params.out2in[out_index].push_back(in_index);
		}
	}
}

void MaxPoolingLayer::init_connection() {
	params.in2out.resize(params.in.size());
	params.out2in.resize(params.out.size());

	for (int c = 0; c < params.in.depth; ++c) {
		for (int y = 0; y < params.out.height; ++y) {
			for (int x = 0; x < params.out.width; ++x) {
				connect_kernel(params.pool_size_x, params.pool_size_y, x, y,
						c);
			}
		}
	}
}

void MaxPoolingLayer::init_backend(BackendType backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params);
	if (static_cast<backend_t>(backend_type) == backend_t::internal
			|| static_cast<backend_t>(backend_type) == backend_t::nnpack
			|| static_cast<backend_t>(backend_type) == backend_t::avx) {
		kernel_fwd.reset(new MaxPoolOp(ctx));
		kernel_back.reset(new MaxPoolGradOp(ctx));
		return;
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}

void MaxPoolingLayer::set_maxpool_params(const shape3d &in, const shape3d &out,
		int pooling_size_x, int pooling_size_y, int stride_x, int stride_y,
		padding pad_type) {
	params.in = in;
	params.out = out;
	params.pool_size_x = pooling_size_x;
	params.pool_size_y = pooling_size_y;
	params.stride_x = stride_x;
	params.stride_y = stride_y;
	params.pad_type = pad_type;
}
}

