/*
 * ConvolutionLayer.h
 *
 *  Created on: Jun 24, 2017
 *      Author: blake
 */

#ifndef INCLUDE_CONVOLUTIONLAYER_H_
#define INCLUDE_CONVOLUTIONLAYER_H_

#include "NeuralLayer.h"
#include "NeuralSignal.h"
#include <AlloyImage.h>
#include <AlloyMath.h>
namespace tgr {

enum class Padding {
	Valid, Same
};
class conv_params: public Params {
public:
	aly::Image1ub tbl;
	aly::dim3 in;
	aly::dim3 in_padded;
	aly::dim3 out;
	aly::dim3 weight;
	bool has_bias;
	Padding pad_type;
	int w_stride;
	int h_stride;
	friend std::ostream &operator<<(std::ostream &o, const conv_params &param) {
		o << "in:        " << param.in << "\n";
		o << "out:       " << param.out << "\n";
		o << "in_padded: " << param.in_padded << "\n";
		o << "weight:    " << param.weight << "\n";
		o << "has_bias:  " << param.has_bias << "\n";
		o << "w_stride:  " << param.w_stride << "\n";
		o << "h_stride:  " << param.h_stride << "\n";
		return o;
	}
};
class Conv2dPadding {
 public:
  Conv2dPadding() {}
  Conv2dPadding(const conv_params &params) : params_(params) {}

  /* Applies padding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with padding applied
   */
  void copy_and_pad_input(const Tensor &in, Tensor &out) {
    if (params_.pad_type == Padding::Valid) {
      return;
    }

    Tensor buf(in.size());

    for_i(true, buf.size(), [&](int sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in_padded.size());

      // make padded version in order to avoid corner-case in fprop/bprop
      for (int c = 0; c < params_.in.z; c++) {
        float_t *pimg = &buf[sample][params_.in_padded(params_.weight.x / 2, params_.weight.y / 2, c)];
        const float_t *pin = &in[sample][params_.in(0, 0, c)];
        for (int y = 0; y < params_.in.y; y++) {
          std::copy(pin, pin + params_.in.x, pimg);
          pin += params_.in.x;
          pimg += params_.in_padded.x;
        }
      }
    });

    // shrink buffer to output
    out = buf;
  }

  /* Applies unpadding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with unpadding applied
   */
  void copy_and_unpad_delta(const Tensor &delta, Tensor &delta_unpadded) {
    if (params_.pad_type == Padding::Valid) {
      return;
    }

    Tensor buf(delta.size());

    for_i(true, buf.size(), [&](int sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in.size());
      for (int c = 0; c < params_.in.z; c++) {
        const float_t *pin = &delta[sample][params_.in_padded(params_.weight.x / 2, params_.weight.y / 2, c)];
        float_t *pdst = &buf[sample][params_.in(0, 0, c)];
        for (int y = 0; y < params_.in.y; y++) {
          std::copy(pin, pin + params_.in.x, pdst);
          pdst += params_.in.x;
          pin += params_.in_padded.x;
        }
      }
    });

    // shrink buffer to output
    delta_unpadded = buf;
  }

 private:
  conv_params params_;
};
class ConvolutionLayer: public NeuralLayer {
	conv_params params_;
	static int conv_out_length(int in_length, int window_size, int stride,
			Padding pad_type) const {
		int output_length;
		if (pad_type == Padding::Same) {
			output_length = in_length;
		} else if (pad_type == Padding::Valid) {
			output_length = in_length - window_size + 1;
		} else {
			throw std::runtime_error("Not recognized pad_type.");
		}
		return (output_length + stride - 1) / stride;
	}
	int in_length(int in_length, int window_size, Padding pad_type) const {
		return pad_type == Padding::Same ?
				(in_length + window_size - 1) : in_length;
	}
	static int conv_out_dim(int in_width, int in_height, int window_size,
			int w_stride, int h_stride, Padding pad_type) {
		return conv_out_length(in_width, window_size, w_stride, pad_type)
				* conv_out_length(in_height, window_size, h_stride, pad_type);
	}

	int conv_out_dim(int in_width, int in_height, int window_width,
			int window_height, int w_stride, int h_stride,
			Padding pad_type) const {
		return conv_out_length(in_width, window_width, w_stride, pad_type)
				* conv_out_length(in_height, window_height, h_stride, pad_type);
	}
	void conv_set_params(const aly::int3 &in, int w_width, int w_height,
			int outc, Padding ptype, bool has_bias, int w_stride, int h_stride,
			const aly::Image1ub& tbl = aly::Image1ub()) {
		params_.in = in;
		params_.in_padded = aly::int3(in_length(in.x, w_width, ptype),
				in_length(in.y, w_height, ptype), in.z);
		params_.out = aly::int3(conv_out_length(in.x, w_width, w_stride, ptype),
				conv_out_length(in.y, w_height, h_stride, ptype), outc);
		params_.weight = aly::int3(w_width, w_height, in.z * outc);
		params_.has_bias = has_bias;
		params_.pad_type = ptype;
		params_.w_stride = w_stride;
		params_.h_stride = h_stride;
		params_.tbl = tbl;

		// init Padding buffer
		if (params_.pad_type == Padding::Same) {
			cws_.prev_delta_padded_.resize(1,Storage(params_.in_padded.size(), float_t(0)));
		}

		// set parameters to Padding operation
		padding_op_ = Conv2dPadding(params_);
	}
	ConvolutionLayer(int in_width, int in_height, int window_width,
			int window_height, int in_channels, int out_channels,
			const aly::Image1ub& connection_table, Padding pad_type =
					Padding::Valid, bool has_bias = true, int w_stride = 1,
			int h_stride = 1, BackendType backend_type = DefaultEngine()) :
			NeuralLayer("Convolution", ChannelOrder(has_bias), {
					ChannelType::data }) {
		conv_set_params(aly::int3(in_width, in_height, in_channels),
				window_width, window_height, out_channels, pad_type, has_bias,
				w_stride, h_stride, connection_table);
		init_backend(backend_type);
		setBackendType(backend_type);
	}
};

}

#endif /* INCLUDE_CONVOLUTIONLAYER_H_ */
