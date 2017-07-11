/*
 * GlobalAveragePoolingLayer.h
 *
 *  Created on: Jul 11, 2017
 *      Author: blake
 */

#ifndef INCLUDE_GLOBALAVERAGEPOOLINGLAYER_H_
#define INCLUDE_GLOBALAVERAGEPOOLINGLAYER_H_
#include "NeuralLayer.h"

#include "tiny_dnn/tiny_dnn.h"


namespace tgr {
class GlobalAveragePoolingLayer: public NeuralLayer {
public:
	/**
	 * @param in_width     [in] width of input image
	 * @param in_height    [in] height of input image
	 * @param in_channels  [in] the number of input image channels (depth)
	 **/
	GlobalAveragePoolingLayer(int in_width, int in_height, int in_channels,
			BackendType backend_type = DefaultEngine());
	GlobalAveragePoolingLayer(const aly::dim3& in_shape,BackendType backend_type = DefaultEngine()) :
			GlobalAveragePoolingLayer(in_shape.x, in_shape.y, in_shape.z,
					backend_type) {
	}
	// move constructor
	GlobalAveragePoolingLayer(GlobalAveragePoolingLayer &&other);
	virtual int getFanInSize() const override {
		return static_cast<int>(params.in.width * params.in.height);
	}
	virtual int getFanOutSize() const override {
		return 1;
	}
	void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	std::vector<aly::dim3> getInputDimensions() const override;
	std::vector<aly::dim3> getOutputDimensions() const override;
	std::pair<int, int> pool_size() const;
private:
	tiny_dnn::core::global_avepool_params params;

	/* forward op context */
	tiny_dnn::core::OpKernelContext fwd_ctx;

	/* backward op context */
	tiny_dnn::core::OpKernelContext bwd_ctx;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back;
	void init_backend(tiny_dnn::core::backend_t backend_type);
	void set_global_avepool_params(const aly::dim3&in, const aly::dim3&out);

}
;
}

#endif /* INCLUDE_GLOBALAVERAGEPOOLINGLAYER_H_ */
