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
#include <AlloyImage.h>
#include <AlloyMath.h>
namespace tgr {
class ConvolutionLayer: public NeuralLayer {
public:
	ConvolutionLayer(int in_width, int in_height, int window_width,
			int window_height, int in_channels, int out_channels,
			const tiny_dnn::core::ConnectionTable& connection_table,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine());
	ConvolutionLayer(int in_width, int in_height, int window_size, int in_channels, int out_channels,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine()):ConvolutionLayer(in_width,
                          in_height,
                          window_size,
                          window_size,
                          in_channels,
                          out_channels,
                          tiny_dnn::core::ConnectionTable(),
                          pad_type,
                          has_bias,
                          w_stride,
                          h_stride,
                          backend_type){

	}
	ConvolutionLayer(int in_width, int in_height, int window_width,int window_height, int in_channels, int out_channels,
			Padding pad_type = Padding::Valid, bool has_bias = true,
			int w_stride = 1, int h_stride = 1, BackendType backend_type =
					DefaultEngine()):ConvolutionLayer(in_width,
                          in_height,
                          window_width,
                          window_height,
                          in_channels,
                          out_channels,
                          tiny_dnn::core::ConnectionTable(),
                          pad_type,
                          has_bias,
                          w_stride,
                          h_stride,
                          backend_type){
	}
	virtual void getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual void getStencilWeight(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual bool getStencilBias(const aly::int3& pos,aly::int3& stencil) const override;
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
	tiny_dnn::core::conv_params params;

	/* Padding operation */
	tiny_dnn::core::Conv2dPadding padding_op;

	/* forward op context */
	tiny_dnn::core::OpKernelContext fwd_ctx;
	/* backward op context */
	tiny_dnn::core::OpKernelContext bwd_ctx;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back;
	std::vector<Tensor *> fwd_in_data;
	std::vector<Tensor *> bwd_in_data;
	std::vector<Tensor *> bwd_in_grad;
	/* Buffer to store padded data */
	struct conv_layer_worker_specific_storage {
		Tensor prev_out_padded;
		Tensor prev_delta_padded;
	} cws_;
	Tensor* in_data_padded(const std::vector<Tensor*> &in);
	void conv_set_params(const tiny_dnn::shape3d &in, int w_width, int w_height, int outc,
			tiny_dnn::padding ptype, bool has_bias, int w_stride, int h_stride,
			const ConnectionTable &tbl = ConnectionTable());

	int in_length(int in_length, int window_size,
			tiny_dnn::padding pad_type) const;
	static int conv_out_dim(int in_width, int in_height, int window_size,
			int w_stride, int h_stride, Padding pad_type);
	void init_backend(const backend_t backend_type);
	int conv_out_dim(int in_width, int in_height, int window_width,
			int window_height, int w_stride, int h_stride,
			Padding pad_type) const;
};
typedef std::shared_ptr<ConvolutionLayer> ConvolutionLayerPtr;

}

#endif /* INCLUDE_CONVOLUTIONLAYER_H_ */
