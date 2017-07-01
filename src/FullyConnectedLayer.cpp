/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "FullyConnectedLayer.h"
#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"
#include "tiny_dnn/core/kernels/fully_connected_op.h"

using namespace tiny_dnn;
namespace tgr {
/**
 * @param in_dim [in] number of elements of the input
 * @param out_dim [in] number of elements of the output
 * @param has_bias [in] whether to include additional bias to the layer
 **/
FullyConnectedLayer::FullyConnectedLayer(int in_dim, int out_dim, bool has_bias,
		BackendType backend_type) :
		NeuralLayer("Fully Connected", ChannelOrder(has_bias), {
				ChannelType::data }) {
	set_params(in_dim, out_dim, has_bias);
	init_backend(static_cast<backend_t>(backend_type));
	NeuralLayer::setBackendType(backend_type);
}

// move constructor
FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayer &&other) :
		NeuralLayer(std::move(other)), params_(std::move(other.params_)), kernel_fwd_(
				std::move(other.kernel_fwd_)), kernel_back_(
				std::move(other.kernel_back_)) {
	init_backend(static_cast<backend_t>(other.getBackendType()));
}

void FullyConnectedLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	// forward fully connected op context
	fwd_ctx_.set_in_out(in_data, out_data);
	fwd_ctx_.setParallelize(NeuralLayer::parallelize);
	fwd_ctx_.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch fully connected kernel
	kernel_fwd_->compute(fwd_ctx_);
}

void FullyConnectedLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	// backward fully connected op context
	bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
	bwd_ctx_.setParallelize(NeuralLayer::parallelize);
	bwd_ctx_.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch fully connected kernel
	kernel_back_->compute(bwd_ctx_);
}

void FullyConnectedLayer::set_params(const int in_size, const int out_size,
		bool has_bias) {
	params_.in_size_ = in_size;
	params_.out_size_ = out_size;
	params_.has_bias_ = has_bias;
}

void FullyConnectedLayer::init_backend(tiny_dnn::core::backend_t backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params_);

	if (backend_type == backend_t::internal || backend_type == backend_t::avx
			|| backend_type == backend_t::nnpack) {
		kernel_fwd_.reset(new FullyConnectedOp(ctx));
		kernel_back_.reset(new FullyConnectedGradOp(ctx));
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}
}

