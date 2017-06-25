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

int ConvolutionLayer::conv_out_dim(int in_width, int in_height, int window_size,
		int w_stride, int h_stride, Padding pad_type) {
	return tiny_dnn::conv_out_length(in_width, window_size, w_stride,
			static_cast<tiny_dnn::padding>(pad_type))
			* tiny_dnn::conv_out_length(in_height, window_size, h_stride,
					static_cast<tiny_dnn::padding>(pad_type));
}
void ConvolutionLayer::conv_set_params(const shape3d &in, int w_width,
		int w_height, int outc, padding ptype, bool has_bias, int w_stride,
		int h_stride, const connection_table &tbl) {
	params_.in = in;
	params_.in_padded = shape3d(in_length(in.width_, w_width, ptype),
			in_length(in.height_, w_height, ptype), in.depth_);
	params_.out = shape3d(conv_out_length(in.width_, w_width, w_stride, ptype),
			conv_out_length(in.height_, w_height, h_stride, ptype), outc);
	params_.weight = shape3d(w_width, w_height, in.depth_ * outc);
	params_.has_bias = has_bias;
	params_.pad_type = ptype;
	params_.w_stride = w_stride;
	params_.h_stride = h_stride;
	params_.tbl = tbl;

	// init padding buffer
	if (params_.pad_type == padding::same) {
		cws_.prev_delta_padded_.resize(1,
				vec_t(params_.in_padded.size(), float_t(0)));
	}

	// set parameters to padding operation
	padding_op_ = Conv2dPadding(params_);
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
			NeuralLayer::device(), &params_);
	if (backend_type == backend_t::internal || backend_type == backend_t::nnpack
			|| backend_type == backend_t::avx) {
		kernel_fwd_.reset(new Conv2dOp(ctx));
		kernel_back_.reset(new Conv2dGradOp(ctx));
		return;
	} else if (backend_type == backend_t::opencl) {
		throw nn_error("Not implemented engine: " + to_string(backend_type));
		/*kernel_fwd_.reset(new Conv2dOpenCLForwardOp(ctx));
		 kernel_back_.reset(new Conv2dOpenCLBackwardOp(ctx));
		 return;*/
	} else if (backend_type == backend_t::libdnn) {
		if (NeuralLayer::device() == nullptr)
			return;
		kernel_fwd_.reset(new Conv2dLibDNNForwardOp(ctx));
		kernel_back_.reset(new Conv2dLibDNNBackwardOp(ctx));
		return;
	} else {
		throw nn_error("Not supported engine: " + to_string(backend_type));
	}
}
ConvolutionLayer::ConvolutionLayer(int in_width, int in_height,
		int window_width, int window_height, int in_channels, int out_channels,
		const core::connection_table& connection_table, Padding pad_type,
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
	if (params_.has_bias) {
		return {Convert(params_.in), Convert(params_.weight),dim3(1, 1, params_.out.depth_)};
	} else {
		return {Convert(params_.in), Convert(params_.weight)};
	}
}
std::vector<aly::dim3> ConvolutionLayer::getOutputDimensions() const {
	return {aly::dim3(params_.out.width_,params_.out.height_,params_.out.depth_)};
}
Tensor* ConvolutionLayer::in_data_padded(const std::vector<Tensor*> &in) {
	return (params_.pad_type == padding::valid) ? in[0] : &cws_.prev_out_padded_;
}
void ConvolutionLayer::forwardPropagation(const std::vector<Tensor*>&in_data,
		std::vector<Tensor*> &out_data) {
	// apply padding to the input tensor
	padding_op_.copy_and_pad_input(*in_data[0], cws_.prev_out_padded_);

	fwd_in_data_.resize(in_data.size());
	std::copy(in_data.begin(), in_data.end(), fwd_in_data_.begin());
	fwd_in_data_[0] = in_data_padded(in_data);

	// forward convolutional op context
	fwd_ctx_.set_in_out(fwd_in_data_, out_data);
	fwd_ctx_.setParallelize(parallelize);
	fwd_ctx_.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch convolutional kernel
	kernel_fwd_->compute(fwd_ctx_);
}
void ConvolutionLayer::backwardPropagation(const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data, std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	bwd_in_data_.resize(in_data.size());
	std::copy(in_data.begin(), in_data.end(), bwd_in_data_.begin());
	bwd_in_data_[0] = in_data_padded(in_data);

	bwd_in_grad_.resize(in_grad.size());
	std::copy(in_grad.begin(), in_grad.end(), bwd_in_grad_.begin());
	if (params_.pad_type == padding::same) {
		bwd_in_grad_[0] = &cws_.prev_delta_padded_;
	}

	bwd_ctx_.set_in_out(bwd_in_data_, out_data, out_grad, bwd_in_grad_);
	bwd_ctx_.setParams(&params_);
	bwd_ctx_.setParallelize(parallelize);
	bwd_ctx_.setEngine(static_cast<backend_t>(NeuralLayer::getBackendType()));

	// launch convolutional kernel
	kernel_back_->compute(bwd_ctx_);

	// unpad deltas
	padding_op_.copy_and_unpad_delta(cws_.prev_delta_padded_, *in_grad[0]);
}
void ConvolutionLayer::setSampleCount(size_t sample_count) {
	NeuralLayer::setSampleCount(sample_count);
	cws_.prev_delta_padded_.resize(sample_count,
			vec_t(params_.in_padded.size(), float_t(0)));
}
int ConvolutionLayer::getFanInSize() const {
	return params_.weight.width_ * params_.weight.height_ * params_.in.depth_;
}
int ConvolutionLayer::getFanOutSize() const {
	return (params_.weight.width_ / params_.w_stride)
			* (params_.weight.height_ / params_.h_stride) * params_.out.depth_;
}

}
