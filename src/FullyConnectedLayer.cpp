/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "FullyConnectedLayer.h"
#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"
#include "tiny_dnn/core/kernels/fully_connected_op.h"
using namespace aly;
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
void FullyConnectedLayer::getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const {
	stencil.resize(params.in_size);
	for (int i = 0; i < (int) stencil.size(); i++) {
		stencil[i] = int3(i, 0, 0);
	}
}
void FullyConnectedLayer::getStencilWeight(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	stencil.resize(params.in_size);
	for (int i = 0; i < (int) stencil.size(); i++) {
		stencil[i] = int3(i, pos.x, 0);
	}
}
bool FullyConnectedLayer::getStencilBias(const aly::int3& pos,
		aly::int3& stencil) const {
	if (params.has_bias) {
		stencil = pos;
		return true;
	} else {
		return false;
	}
}
// move constructor
FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayer &&other) :
		NeuralLayer(std::move(other)), params(std::move(other.params)), kernel_fwd(
				std::move(other.kernel_fwd)), kernel_back(
				std::move(other.kernel_back)) {
	init_backend(static_cast<backend_t>(other.getBackendType()));
}

void FullyConnectedLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	// forward fully connected op context
	fwd_ctx.set_in_out(in_data, out_data);
	fwd_ctx.setParallelize(NeuralLayer::parallelize);
	fwd_ctx.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch fully connected kernel
	kernel_fwd->compute(fwd_ctx);
}

void FullyConnectedLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	// backward fully connected op context
	bwd_ctx.set_in_out(in_data, out_data, out_grad, in_grad);
	bwd_ctx.setParallelize(NeuralLayer::parallelize);
	bwd_ctx.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch fully connected kernel
	kernel_back->compute(bwd_ctx);
}

void FullyConnectedLayer::set_params(const int in_size, const int out_size,
		bool has_bias) {
	params.in_size = in_size;
	params.out_size = out_size;
	params.has_bias = has_bias;
}

void FullyConnectedLayer::init_backend(tiny_dnn::core::backend_t backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params);

	if (backend_type == backend_t::internal || backend_type == backend_t::avx
			|| backend_type == backend_t::nnpack) {
		kernel_fwd.reset(new FullyConnectedOp(ctx));
		kernel_back.reset(new FullyConnectedGradOp(ctx));
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}
}

