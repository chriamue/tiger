/*
 * DeconvolutionLayer.cpp
 *
 *  Created on: Jun 26, 2017
 *      Author: blake
 */

#include "DeconvolutionLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace aly;
using namespace tiny_dnn;
using namespace tiny_dnn::core;
namespace tgr {
DeconvolutionLayer::DeconvolutionLayer(int in_width, int in_height,
		int window_width, int window_height, int in_channels, int out_channels,
		const tiny_dnn::core::connection_table &connection_table,
		Padding pad_type, bool has_bias, int w_stride, int h_stride,
		BackendType backend_type) :
		NeuralLayer("Deconvolution", ChannelOrder(has_bias),
				{ ChannelType::data }) {
	deconv_set_params(shape3d(in_width, in_height, in_channels), window_width,
			window_height, out_channels, static_cast<padding>(pad_type),
			has_bias, w_stride, h_stride, connection_table);
	init_backend(static_cast<backend_t>(backend_type));
}

///< number of incoming connections for each output unit
int DeconvolutionLayer::getFanInSize() const {
	return params_.weight.width * params_.weight.height * params_.in.depth;
}

///< number of outgoing connections for each input unit
int DeconvolutionLayer::getFanOutSize() const {
	return (params_.weight.width * params_.w_stride)
			* (params_.weight.height * params_.h_stride) * params_.out.depth;
}

void DeconvolutionLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	// launch deconvolutional kernel
	backend_->deconv2d(in_data, out_data);
}

/**
 * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
 *fully-connected layer)
 * @param worker_index id of current worker-task
 * @param in_data      input vectors (same vectors as forward_propagation)
 * @param out_data     output vectors (same vectors as forward_propagation)
 * @param out_grad     gradient of output vectors (i-th vector correspond
 *with
 *out_data[i])
 * @param in_grad      gradient of input vectors (i-th vector correspond
 *with
 *in_data[i])
 **/
void DeconvolutionLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	backend_->deconv2d(in_data, out_data, out_grad, in_grad);
}
std::vector<aly::dim3> DeconvolutionLayer::getInputDimensions() const {
	if (params_.has_bias) {
		return {Convert(params_.in),Convert(params_.weight),
			aly::dim3(1, 1, params_.out.depth)};
	} else {
		return {Convert(params_.in),Convert(params_.weight)};
	}
}
std::vector<aly::dim3> DeconvolutionLayer::getOutputDimensions() const {
	return {Convert(params_.out_unpadded)};
}
void DeconvolutionLayer::init_backend(const backend_t backend_type) {
	std::shared_ptr<core::backend> backend = nullptr;

	// allocate new backend
	if (backend_type == backend_t::internal) {
		backend = std::make_shared<core::tiny_backend>(&params_,
				[this](const Tensor &in) {return copy_and_unpad_output(in);},
				[this](const Tensor &delta, Tensor &dst) {
					return copy_and_pad_delta(delta, dst);
				}, &deconv_layer_worker_storage_);
#ifdef CNN_USE_AVX
	} else if (backend_type == backend_t::avx) {
		backend = std::make_shared<core::avx_backend>(
				&params_,
				[this](const Tensor &in) {return copy_and_unpad_output(in);},
				[this](const Tensor &delta, Tensor &dst) {
					return copy_and_pad_delta(delta, dst);
				},
				&deconv_layer_worker_storage_);
#endif
	} else {
		throw nn_error("Not supported backend type.");
	}

	if (backend) {
		this->backend_ = backend;
		//backend_->set_layer(this); //Blake: What to do here! Create own Backend class?
	} else {
		throw nn_error("Could not allocate the backend.");
	}
}

void DeconvolutionLayer::deconv_set_params(const shape3d &in, int w_width,
		int w_height, int outc, padding ptype, bool has_bias, int w_stride,
		int h_stride, const connection_table &tbl = connection_table()) {
	params_.in = in;
	params_.out = shape3d(deconv_out_length(in.width, w_width, w_stride),
			deconv_out_length(in.height, w_height, h_stride), outc);
	params_.out_unpadded = shape3d(
			deconv_out_unpadded_length(in.width, w_width, w_stride, ptype),
			deconv_out_unpadded_length(in.height, w_height, h_stride, ptype),
			outc);
	params_.weight = shape3d(w_width, w_height, in.depth * outc);
	params_.has_bias = has_bias;
	params_.pad_type = ptype;
	params_.w_stride = w_stride;
	params_.h_stride = h_stride;
	params_.tbl = tbl;

	out2in.resize((size_t) params_.out.area());
	serial_size_t idx = 0;
	for (serial_size_t y = 0; y < params_.in.height; y++) {
		for (serial_size_t x = 0; x < params_.in.width; x++) {
			for (serial_size_t wy = 0; wy < params_.weight.height; wy++) {
				for (serial_size_t wx = 0; wx < params_.weight.width; wx++) {
					size_t index = (y * params_.h_stride + wy)
							* params_.out.width + (x * params_.w_stride + wx);
					out2in[index].push_back(int2(x, y));
				}
			}
		}
	}

}

void DeconvolutionLayer::init_workers(int sample_count) {
	deconv_layer_worker_specific_storage &dws = deconv_layer_worker_storage_;

	if (params_.pad_type == padding::same) {
		dws.curr_out_buf_.resize(sample_count,
				Storage(params_.out_unpadded.size(), float { 0 }));
		dws.curr_delta_padded.resize(sample_count,
				Storage(params_.out.size(), float { 0 }));
	} else {
		dws.curr_out_buf_.clear();
	}
}
int DeconvolutionLayer::in_length(int in_length, int window_size,
		padding pad_type) const {
	return in_length;
}
int DeconvolutionLayer::deconv_out_length(int in_length, int window_size,
		int stride) {
	return (int) ceil((float) (in_length) * stride + window_size - 1);
}
int DeconvolutionLayer::deconv_out_unpadded_length(int in_length,
		int window_size, int stride, padding pad_type) {
	return pad_type == padding::same ?
			(int) ceil((float) in_length * stride) :
			(int) ceil((float) (in_length) * stride + window_size - 1);
}
void DeconvolutionLayer::getStencilInput(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	const std::vector<aly::int2>& out = out2in[pos.x + pos.y * params_.out.width];
	stencil.resize(out.size());
	for (int i = 0; i < out.size(); i++) {
		stencil[i] = int3(out[i], pos.z);
	}
}
void DeconvolutionLayer::getStencilWeight(const aly::int3& pos,
		std::vector<aly::int3>& stencil) const {
	int w = params_.weight.width;
	int h = params_.weight.height;
	stencil.resize(h * w);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			stencil[i] = int3(i, j, pos.z);
		}
	}
}
bool DeconvolutionLayer::getStencilBias(const aly::int3& pos,
		aly::int3& stencil) const {
	if (params_.has_bias) {
		stencil = pos;
		return true;
	} else {
		return false;
	}
}

int DeconvolutionLayer::deconv_out_dim(int in_width, int in_height,
		int window_size, int w_stride, int h_stride, padding pad_type) {
	return deconv_out_unpadded_length(in_width, window_size, w_stride, pad_type)
			* deconv_out_unpadded_length(in_height, window_size, h_stride,
					pad_type);
}

int DeconvolutionLayer::deconv_out_dim(int in_width, int in_height,
		int window_width, int window_height, int w_stride, int h_stride,
		padding pad_type) const {
	return deconv_out_unpadded_length(in_width, window_width, w_stride,
			pad_type)
			* deconv_out_unpadded_length(in_height, window_height, h_stride,
					pad_type);
}

void DeconvolutionLayer::copy_and_pad_delta(const Tensor &delta,
		Tensor &delta_padded) {
	if (params_.pad_type == padding::valid) {
		delta_padded = delta;
	} else {
		for (int sample = 0; sample < delta.size(); sample++) {
			Storage &dst = delta_padded[sample];
			const Storage &src = delta[sample];

			for (int c = 0; c < params_.in.depth; c++) {
				float *pdst = &dst[params_.in.get_index(0, 0, c)];
				const float *pin = &src[params_.in.get_index(0, 0, c)];

				for (int y = 0; y < params_.in.height;
						y++, pdst += params_.in.width, pin += params_.in.width) {
					std::copy(pin, pin + params_.in.width, pdst);
				}
			}
		}
	}
}

void DeconvolutionLayer::copy_and_unpad_output(const Tensor &out) {
	deconv_layer_worker_specific_storage &dws = deconv_layer_worker_storage_;

	dws.curr_out_buf_ = Tensor(out.size(),
			Storage(params_.out_unpadded.size(), 0));
	Tensor *dst_tensor = &dws.curr_out_buf_;

	if (params_.pad_type == padding::valid) {
		dws.curr_out_unpadded_ = &out;
	} else {
		// make unpadded version in order to restore scale in fprop/bprop
		for (int sample = 0; sample < out.size(); sample++) {
			int idx = 0;
			Storage &dst = (*dst_tensor)[sample];
			int wieght_w_half = params_.weight.width / 2;
			int wieght_h_half = params_.weight.height / 2;

			for (int c = 0; c < params_.out_unpadded.depth; c++) {
				float *pimg = &dst[params_.out_unpadded.get_index(0, 0, c)];
				idx = params_.out.get_index(wieght_w_half, wieght_h_half, c);
				const float *pout = &out[sample][idx];

				for (int y = wieght_h_half;
						y < params_.out_unpadded.height + wieght_h_half;
						y++, pout += params_.out.width, pimg +=
								params_.out_unpadded.width) {
					std::copy(pout, pout + params_.out_unpadded.width, pimg);
				}
			}
		}

		dws.curr_out_unpadded_ = &dws.curr_out_buf_;
	}
}

}

