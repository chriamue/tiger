/*
 * InputLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_INPUTLAYER_H_
#define INCLUDE_INPUTLAYER_H_
#include "NeuralLayer.h"
namespace tgr {

class InputLayer: public NeuralLayer {
public:
	InputLayer(const aly::dim3& shape);
	InputLayer(int in_dim);
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
private:
	aly::dim3 shape_;
};
typedef std::shared_ptr<InputLayer> InputLayerPtr;
}

#endif /* INCLUDE_INPUTLAYER_H_ */