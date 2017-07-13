/*
 * MaxPoolingLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_MAXPOOLINGLAYER_H_
#define INCLUDE_MAXPOOLINGLAYER_H_

#include "NeuralLayer.h"

#include "tiny_dnn/tiny_dnn.h"
namespace tgr {
class MaxPoolingLayer: public NeuralLayer {
	MaxPoolingLayer(int in_width, int in_height, int in_channels,
			int pooling_size_x, int pooling_size_y, int stride_x, int stride_y,
			Padding pad_type = Padding::Valid, BackendType backend_type =
					DefaultEngine());
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)
					override;
	virtual int getFanInSize() const override;
	virtual int getFanOutSize() const override;
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void setSampleCount(size_t sample_count) override;
private:
	/* The Max Poling operation params */
	tiny_dnn::core::maxpool_params params;
	std::pair<int, int> pool_size() const;

	/* forward op context */
	tiny_dnn::core::OpKernelContext fwd_ctx;

	/* backward op context */
	tiny_dnn::core::OpKernelContext bwd_ctx;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back;
	void set_maxpool_params(const tiny_dnn::shape3d &in,
			const tiny_dnn::shape3d &out, int pooling_size_x,
			int pooling_size_y, int stride_x, int stride_y,
			tiny_dnn::padding pad_type);
	void connect_kernel(int pooling_size_x, int pooling_size_y, int outx,
			int outy, int c);
	void init_connection();
	void init_backend(BackendType backend_type);
};
typedef std::shared_ptr<MaxPoolingLayer> MaxPoolingLayerPtr;
}

#endif /* INCLUDE_MAXPOOLINGLAYER_H_ */
