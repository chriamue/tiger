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

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/util.h"
#include <AlloyImage.h>
#include <AlloyMath.h>

namespace tgr {
enum class Padding {
	Valid = tiny_dnn::padding::valid, Same = tiny_dnn::padding::same
};
class ConvolutionLayer: public NeuralLayer {
	ConvolutionLayer(int in_width, int in_height, int window_width,
			int window_height, int in_channels, int out_channels,
			const tiny_dnn::core::connection_table& connection_table,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine());
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)
					override;
	virtual void setSampleCount(size_t sample_count) override;
	virtual int getFanInSize() const override;
	virtual int getFanOutSize() const override;
private:
	/* The convolution parameters */
	conv_params params_;

	/* Padding operation */
	Conv2dPadding padding_op_;

	/* forward op context */
	OpKernelContext fwd_ctx_;

	/* backward op context */
	OpKernelContext bwd_ctx_;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd_;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back_;
	std::vector<Tensor *> fwd_in_data_;
	std::vector<Tensor *> bwd_in_data_;
	std::vector<Tensor *> bwd_in_grad_;
	/* Buffer to store padded data */
	struct conv_layer_worker_specific_storage {
		Tensor prev_out_padded_;
		Tensor prev_delta_padded_;
	} cws_;
	Tensor* in_data_padded(const std::vector<Tensor*> &in);
	void conv_set_params(const tiny_dnn::shape3d &in, int w_width, int w_height, int outc,
			tiny_dnn::padding ptype, bool has_bias, int w_stride, int h_stride,
			const connection_table &tbl = connection_table());
	int in_length(int in_length, int window_size,
			tiny_dnn::padding pad_type) const;
	static int conv_out_dim(int in_width, int in_height, int window_size,
			int w_stride, int h_stride, Padding pad_type);
	void init_backend(const backend_t backend_type);
	int conv_out_dim(int in_width, int in_height, int window_width,
			int window_height, int w_stride, int h_stride,
			Padding pad_type) const;
};

}

#endif /* INCLUDE_CONVOLUTIONLAYER_H_ */
