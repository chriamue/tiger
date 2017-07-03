/*
 * DeconvolutionLayer.h
 *
 *  Created on: Jun 26, 2017
 *      Author: blake
 */

#ifndef INCLUDE_DECONVOLUTIONLAYER_H_
#define INCLUDE_DECONVOLUTIONLAYER_H_

#include "NeuralLayer.h"
#include "NeuralSignal.h"
#include "tiny_dnn/tiny_dnn.h"
namespace tgr {
class DeconvolutionLayer: public NeuralLayer {
public:
	/**
	 * constructing deconvolutional layer
	 *
	 * @param in_width         [in] input image width
	 * @param in_height        [in] input image height
	 * @param window_width     [in] window_width(kernel) size of convolution
	 * @param window_height    [in] window_height(kernel) size of convolution
	 * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
	 * @param out_channels     [in] output image channels
	 * @param connection_table [in] definition of connections between in-channels
	 *and out-channels
	 * @param pad_type         [in] rounding strategy
	 *                               valid: use valid pixels of input only.
	 *output-size = (in-width - window_size + 1) *
	 *(in-height - window_size + 1) * out_channels
	 *                               same: add zero-padding to keep same
	 *width/height. output-size = in-width * in-height *
	 *out_channels
	 * @param has_bias         [in] whether to add a bias vector to the filter
	 *outputs
	 * @param w_stride         [in] specify the horizontal interval at which to
	 *apply the filters to the input
	 * @param h_stride         [in] specify the vertical interval at which to
	 *apply
	 *the filters to the input
	 **/
	DeconvolutionLayer(int in_width, int in_height, int window_width,
			int window_height, int in_channels, int out_channels,
			const tiny_dnn::core::ConnectionTable &connection_table,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine());

	DeconvolutionLayer(int in_width, int in_height, int window_size,
			int in_channels, int out_channels,
			const tiny_dnn::core::ConnectionTable &connection_table,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine()) :
			DeconvolutionLayer(in_width, in_height, window_size, window_size,
					in_channels, out_channels, connection_table, pad_type,
					has_bias, w_stride, h_stride, backend_type) {

	}

	///< number of incoming connections for each output unit
	virtual int getFanInSize() const override;

	///< number of outgoing connections for each input unit
	virtual int getFanOutSize() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
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
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad);
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual void getStencilWeight(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual bool getStencilBias(const aly::int3& pos,aly::int3& stencil) const override;
private:
	void init_backend(const tiny_dnn::core::backend_t backend_type);
	void deconv_set_params(const tiny_dnn::shape3d &in, int w_width,
			int w_height, int outc, tiny_dnn::padding ptype, bool has_bias,
			int w_stride, int h_stride,
			const tiny_dnn::core::ConnectionTable &tbl);
	void init_workers(int sample_count);
	int in_length(int in_length, int window_size,
			tiny_dnn::padding pad_type) const;
	static int deconv_out_length(int in_length, int window_size, int stride);
	static int deconv_out_unpadded_length(int in_length, int window_size,
			int stride, tiny_dnn::padding pad_type);
	static int deconv_out_dim(int in_width, int in_height, int window_size,
			int w_stride, int h_stride, tiny_dnn::padding pad_type);
	int deconv_out_dim(int in_width, int in_height, int window_width,
			int window_height, int w_stride, int h_stride,
			tiny_dnn::padding pad_type) const;
	void copy_and_pad_delta(const Tensor &delta, Tensor &delta_padded);
	void copy_and_unpad_output(const Tensor &out);
	/* The convolution parameters */
	std::vector<std::vector<aly::int2>> out2in;
	tiny_dnn::core::deconv_params params;
	std::shared_ptr<tiny_dnn::core::backend> backend;
	tiny_dnn::core::deconv_layer_worker_specific_storage deconv_layer_worker_storage;
};
typedef std::shared_ptr<DeconvolutionLayer> DeconvolutionLayerPtr;
}

#endif /* INCLUDE_DECONVOLUTIONLAYER_H_ */
