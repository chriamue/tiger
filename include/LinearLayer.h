/*
 * LinearLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_LINEARLAYER_H_
#define INCLUDE_LINEARLAYER_H_

#include "NeuralLayer.h"
/**
 * element-wise operation: ```f(x) = h(scale*x+bias)```
 */
namespace tgr {
class LinearLayer: public NeuralLayer {
public:
	/**
	 * @param dim   [in] number of elements
	 * @param scale [in] factor by which to multiply
	 * @param bias  [in] bias term
	 **/
	LinearLayer(int dim, float scale = 1.0f, float bias = 0.0f);

	virtual std::vector<aly::dim3> getInputDimensions() const override;

	virtual std::vector<aly::dim3> getOutputDimensions() const override;

	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;

	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
protected:
	int dim_;
	float scale_, bias_;
}
;
}
#endif /* INCLUDE_LINEARLAYER_H_ */
