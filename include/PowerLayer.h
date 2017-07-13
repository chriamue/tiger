/*
 * PowerLayer.h
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#ifndef INCLUDE_POWERLAYER_H_
#define INCLUDE_POWERLAYER_H_

#include "NeuralLayer.h"
namespace tgr {
/**
 * element-wise pow: ```y = scale*x^factor```
 **/
class PowerLayer: public NeuralLayer {
public:
	/**
	 * @param in_shape [in] shape of input tensor
	 * @param factor   [in] floating-point number that specifies a power
	 * @param scale    [in] scale factor for additional multiply
	 */
	PowerLayer(const aly::dim3 &in_shape, float factor, float scale = 1.0f);

	/**
	 * @param prev_layer [in] previous layer to be connected
	 * @param factor     [in] floating-point number that specifies a power
	 * @param scale      [in] scale factor for additional multiply
	 */
	PowerLayer(const NeuralLayer &prev_layer, float factor, float scale = 1.0f);
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	float getFactor() const {
		return factor;
	}
	float getScale() const {
		return scale;
	}
private:
	aly::dim3 in_shape;
	float factor;
	float scale;
};

}

#endif /* INCLUDE_POWERLAYER_H_ */
