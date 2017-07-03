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
		return params.in_size;
	}

	virtual int getFanOutSize() const override {
		return params.out_size;
	}

	virtual std::vector<aly::dim3> getInputDimensions() const override {
		if (params.has_bias) {
			return {aly::dim3(params.in_size, 1, 1),
				aly::dim3(params.in_size, params.out_size, 1),
				aly::dim3(params.out_size, 1, 1)};
		} else {
			return {aly::dim3(params.in_size, 1, 1),
				aly::dim3(params.in_size, params.out_size, 1)};
		}
	}

	virtual std::vector<aly::dim3> getOutputDimensions() const override {
		return {aly::dim3(params.out_size, 1, 1)};
	}

	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;

	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	virtual void getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual void getStencilWeight(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual bool getStencilBias(const aly::int3& pos,aly::int3& stencil) const override;
protected:
	void set_params(const int in_size, const int out_size, bool has_bias);
	void init_backend(tiny_dnn::core::backend_t backend_type);

private:
	/* The layer parameters */
	tiny_dnn::core::fully_params params;

	/* forward op context */
	tiny_dnn::core::OpKernelContext fwd_ctx;

	/* backward op context */
	tiny_dnn::core::OpKernelContext bwd_ctx;

	/* Forward and backward ops */
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_fwd;
	std::shared_ptr<tiny_dnn::core::OpKernel> kernel_back;
};
typedef std::shared_ptr<FullyConnectedLayer> FullyConnectedLayerPtr;
}

#endif /* INCLUDE_FULLYCONNECTEDLAYER_H_ */
