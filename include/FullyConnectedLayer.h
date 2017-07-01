/*
 * FullyConnectedLayer.h
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#ifndef INCLUDE_FULLYCONNECTEDLAYER_H_
#define INCLUDE_FULLYCONNECTEDLAYER_H_

#include "NeuralLayer.h"
#include "tiny_dnn/tiny_dnn.h"
namespace tgr {

/**
 * compute fully-connected(matmul) operation
 **/
class FullyConnectedLayer: public NeuralLayer {
public:
	/**
	 * @param in_dim [in] number of elements of the input
	 * @param out_dim [in] number of elements of the output
	 * @param has_bias [in] whether to include additional bias to the layer
	 **/
	FullyConnectedLayer(int in_dim, int out_dim, bool has_bias = true,
			BackendType backend_type = DefaultEngine());
	// move constructor
	FullyConnectedLayer(FullyConnectedLayer &&other);

	virtual int getFanInSize() const override {
		return params_.in_size_;
	}

	virtual int getFanOutSize() const override {
		return params_.out_size_;
	}

	virtual std::vector<aly::dim3> getInputDimensions() const override {
		if (params_.has_bias_) {
			return {aly::dim3(params_.in_size_, 1, 1),
				aly::dim3(params_.in_size_, params_.out_size_, 1),
				aly::dim3(params_.out_size_, 1, 1)};
		} else {
			return {aly::dim3(params_.in_size_, 1, 1),
				aly::dim3(params_.in_size_, params_.out_size_, 1)};
		}
	}

	virtual std::vector<aly::dim3> getOutputDimensions() const override {
		return {aly::dim3(params_.out_size_, 1, 1)};
	}

	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;

	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;

protected:
	void set_params(const int in_size, const int out_size, bool has_bias);
	void init_backend(tiny_dnn::core::backend_t backend_type);

private:
	/* The layer parameters */
	tiny_dnn::core::fully_params params_;

	/* forward op context */
	tiny_dnn::core::OpKernelContext fwd_ctx_;

	/* backward op context */
	tiny_dnn::core::OpKernelContext bwd_ctx_;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd_;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back_;
};
typedef std::shared_ptr<FullyConnectedLayer> FullyConnectedLayerPtr;
}

#endif /* INCLUDE_FULLYCONNECTEDLAYER_H_ */
