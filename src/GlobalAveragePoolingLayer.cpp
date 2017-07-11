/*
 * GlobalAveragePoolingLayer.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: blake
 */

#include "GlobalAveragePoolingLayer.h"
using namespace tiny_dnn;
namespace tgr {
GlobalAveragePoolingLayer::GlobalAveragePoolingLayer(int in_width,
		int in_height, int in_channels, BackendType backend_type) :
		NeuralLayer("Global Average Pooling", { ChannelType::data }, {
				ChannelType::data }) {
	set_global_avepool_params(aly::dim3(in_width, in_height, in_channels),
			aly::dim3(in_channels, 1, 1));
	init_backend(static_cast<core::backend_t>(backend_type));
}

// move constructor
GlobalAveragePoolingLayer::GlobalAveragePoolingLayer(
		GlobalAveragePoolingLayer && other) :
		NeuralLayer(std::move(other)), params(std::move(other.params)) {
	init_backend(static_cast<core::backend_t>(getBackendType()));
}

void GlobalAveragePoolingLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	fwd_ctx.set_in_out(in_data, out_data);
	fwd_ctx.setParallelize(parallelize);
	fwd_ctx.setEngine(static_cast<core::backend_t>(getBackendType()));
	kernel_fwd->compute(fwd_ctx);
}
void GlobalAveragePoolingLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	bwd_ctx.set_in_out(in_data, out_data, out_grad, in_grad);
	bwd_ctx.setParallelize(parallelize);
	bwd_ctx.setEngine(static_cast<core::backend_t>(getBackendType()));
	kernel_back->compute(bwd_ctx);
}
std::vector<aly::dim3> GlobalAveragePoolingLayer::getInputDimensions() const {
	return std::vector<aly::dim3> { Convert(params.in) };
}
std::vector<aly::dim3> GlobalAveragePoolingLayer::getOutputDimensions() const {
	return std::vector<aly::dim3> { Convert(params.out) };
}
std::pair<int, int> GlobalAveragePoolingLayer::pool_size() const {
	return std::make_pair(params.in.width, params.in.height);
}
void GlobalAveragePoolingLayer::init_backend(
		tiny_dnn::core::backend_t backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params);
	setBackendType(static_cast<BackendType>(backend_type));
	if (backend_type == backend_t::internal || backend_type == backend_t::avx
			|| backend_type == backend_t::nnpack) {
		kernel_fwd.reset(new GlobalAvePoolOp(ctx));
		kernel_back.reset(new GlobalAvePoolGradOp(ctx));
		return;
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}
void GlobalAveragePoolingLayer::set_global_avepool_params(const aly::dim3&in,
		const aly::dim3&out) {
	params.in = Convert(in);
	params.out = Convert(out);
}

}

