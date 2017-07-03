/*
 * ConvolutiondnnLayer.cpp
 *
 *  Created on: Jun 24, 2017
 *      Author: blake
 */
#include "ConvolutionLayer.h"
using namespace tiny_dnn;
using namespace tiny_dnn::core;
using namespace aly;
namespace tgr {
void ConvolutionLayer::getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const {
	int w = params.weight.width;
	int h = params.weight.height;
	int inw = params.in.width;
	int inh = params.in.height;
	int lowi = (params.pad_type == padding::valid) ? pos.x : pos.x - w / 2;
	int lowj = (params.pad_type == padding::valid) ? pos.y : pos.y - h / 2;
	int hii = (params.pad_type == padding::valid) ? pos.x + w : lowi + w;
	int hij = (params.pad_type == padding::valid) ? pos.y + h : lowj + h;
	for (int j = pos.y; j < hij; j += params.w_stride) {
		for (int i = pos.x; i < hij; i += params.h_stride) {
			if (i > 0 && j > 0 && i < inw && j < inh) {
				stencil.push_back(int3(i, j, pos.z));
			}
		}
	}
}

void ConvolutionLayer::getStencilWeight(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	int w = params.weight.width;
	int h = params.weight.height;
	stencil.resize(h * w);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			stencil[i] = int3(i, j, pos.z);
		}
	}
}
bool ConvolutionLayer::getStencilBias(const aly::int3& pos,
		aly::int3& stencil) const {
	if (params.has_bias) {
		stencil = pos;
		return true;
	} else {
		return false;
	}
}
int ConvolutionLayer::conv_out_dim(int in_width, int in_height, int window_size,
		int w_stride, int h_stride, Padding pad_type) {
	return tiny_dnn::conv_out_length(in_width, window_size, w_stride,
			static_cast<tiny_dnn::padding>(pad_type))
			* tiny_dnn::conv_out_length(in_height, window_size, h_stride,
					static_cast<tiny_dnn::padding>(pad_type));
}
void ConvolutionLayer::conv_set_params(const shape3d &in, int w_width,
		int w_height, int outc, padding ptype, bool has_bias, int w_stride,
		int h_stride, const ConnectionTable &tbl) {
	params.in = in;
	params.in_padded = shape3d(in_length(in.width, w_width, ptype),
			in_length(in.height, w_height, ptype), in.depth);
	params.out = shape3d(conv_out_length(in.width, w_width, w_stride, ptype),
			conv_out_length(in.height, w_height, h_stride, ptype), outc);
	params.weight = shape3d(w_width, w_height, in.depth * outc);
	params.has_bias = has_bias;
	params.pad_type = ptype;
	params.w_stride = w_stride;
	params.h_stride = h_stride;
	params.tbl = tbl;

	// init padding buffer
	if (params.pad_type == padding::same) {
		cws_.prev_delta_padded.resize(1,
				vec_t(params.in_padded.size(), float_t(0)));
	}

	// set parameters to padding operation
	padding_op = Conv2dPadding(params);
}

int ConvolutionLayer::in_length(int in_length, int window_size,
		tiny_dnn::padding pad_type) const {
	return pad_type == tiny_dnn::padding::same ?
			(in_length + window_size - 1) : in_length;
}
int ConvolutionLayer::conv_out_dim(int in_width, int in_height,
		int window_width, int window_height, int w_stride, int h_stride,
		Padding pad_type) const {
	return tiny_dnn::conv_out_length(in_width, window_width, w_stride,
			static_cast<padding>(pad_type))
			* tiny_dnn::conv_out_length(in_height, window_height, h_stride,
					static_cast<padding>(pad_type));
}
void ConvolutionLayer::init_backend(const backend_t backend_type) {
	core::OpKernelConstruction ctx = core::OpKernelConstruction(
			NeuralLayer::device(), &params);
	if (backend_type == backend_t::internal || backend_type == backend_t::nnpack
			|| backend_type == backend_t::avx) {
		kernel_fwd.reset(new Conv2dOp(ctx));
		kernel_back.reset(new Conv2dGradOp(ctx));
		return;
	} else if (backend_type == backend_t::opencl) {
		throw nn_error("Not implemented engine: " + to_string(backend_type));
		/*kernel_fwd_.reset(new Conv2dOpenCLForwardOp(ctx));
		 kernel_back_.reset(new Conv2dOpenCLBackwardOp(ctx));
		 return;*/
	} else if (backend_type == backend_t::libdnn) {
		if (NeuralLayer::device() == nullptr)
			return;
		kernel_fwd.reset(new Conv2dLibDNNForwardOp(ctx));
		kernel_back.reset(new Conv2dLibDNNBackwardOp(ctx));
		return;
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}
ConvolutionLayer::ConvolutionLayer(int in_width, int in_height,
		int window_width, int window_height, int in_channels, int out_channels,
		const core::ConnectionTable& connection_table, Padding pad_type,
		bool has_bias, int w_stride, int h_stride, BackendType backend_type) :
		NeuralLayer("Convolution", ChannelOrder(has_bias),
				{ ChannelType::data }) {
	conv_set_params(shape3d(in_width, in_height, in_channels), window_width,
			window_height, out_channels, static_cast<padding>(pad_type),
			has_bias, w_stride, h_stride, connection_table);
	init_backend(static_cast<backend_t>(backend_type));
	setBackendType(backend_type);
}
std::vector<aly::dim3> ConvolutionLayer::getInputDimensions() const {
	if (params.has_bias) {
		return {Convert(params.in), Convert(params.weight),dim3(1, 1, params.out.depth)};
	} else {
		return {Convert(params.in), Convert(params.weight)};
	}
}
std::vector<aly::dim3> ConvolutionLayer::getOutputDimensions() const {
	return {aly::dim3(params.out.width,params.out.height,params.out.depth)};
}
Tensor* ConvolutionLayer::in_data_padded(const std::vector<Tensor*> &in) {
	return (params.pad_type == padding::valid) ? in[0] : &cws_.prev_out_padded;
}
void ConvolutionLayer::forwardPropagation(const std::vector<Tensor*>&in_data,
		std::vector<Tensor*> &out_data) {
	// apply padding to the input tensor
	padding_op.copy_and_pad_input(*in_data[0], cws_.prev_out_padded);

	fwd_in_data.resize(in_data.size());
	std::copy(in_data.begin(), in_data.end(), fwd_in_data.begin());
	fwd_in_data[0] = in_data_padded(in_data);

	// forward convolutional op context
	fwd_ctx.set_in_out(fwd_in_data, out_data);
	fwd_ctx.setParallelize(parallelize);
	fwd_ctx.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch convolutional kernel
	kernel_fwd->compute(fwd_ctx);
}
void ConvolutionLayer::backwardPropagation(const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data, std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	bwd_in_data.resize(in_data.size());
	std::copy(in_data.begin(), in_data.end(), bwd_in_data.begin());
	bwd_in_data[0] = in_data_padded(in_data);

	bwd_in_grad.resize(in_grad.size());
	std::copy(in_grad.begin(), in_grad.end(), bwd_in_grad.begin());
	if (params.pad_type == padding::same) {
		bwd_in_grad[0] = &cws_.prev_delta_padded;
	}

	bwd_ctx.set_in_out(bwd_in_data, out_data, out_grad, bwd_in_grad);
	bwd_ctx.setParams(&params);
	bwd_ctx.setParallelize(parallelize);
	bwd_ctx.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch convolutional kernel
	kernel_back->compute(bwd_ctx);

	// unpad deltas
	padding_op.copy_and_unpad_delta(cws_.prev_delta_padded, *in_grad[0]);
}
void ConvolutionLayer::setSampleCount(size_t sample_count) {
	NeuralLayer::setSampleCount(sample_count);
	cws_.prev_delta_padded.resize(sample_count,
			vec_t(params.in_padded.size(), float_t(0)));
}
int ConvolutionLayer::getFanInSize() const {
	return params.weight.width * params.weight.height * params.in.depth;
}
int ConvolutionLayer::getFanOutSize() const {
	return (params.weight.width / params.w_stride)
			* (params.weight.height / params.h_stride) * params.out.depth;
}

}
