/*
 * PowerLayer.cpp
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#include "PowerLayer.h"
namespace tgr {
/**
 * @param in_shape [in] shape of input tensor
 * @param factor   [in] floating-point number that specifies a power
 * @param scale    [in] scale factor for additional multiply
 */
PowerLayer::PowerLayer(const aly::dim3 &in_shape, float factor, float scale) :
		NeuralLayer("Power", { ChannelType::data }, { ChannelType::data }), in_shape(
				in_shape), factor(factor), scale(scale) {
}
/**
 * @param prev_layer [in] previous layer to be connected
 * @param factor     [in] floating-point number that specifies a power
 * @param scale      [in] scale factor for additional multiply
 */
PowerLayer::PowerLayer(const NeuralLayer &prev_layer, float factor, float scale) :
		NeuralLayer("Power", { ChannelType::data }, { ChannelType::data }), in_shape(
				prev_layer.getOutputDimensions()[0]), factor(factor), scale(
				scale) {
}
std::vector<aly::dim3> PowerLayer::getInputDimensions() const {
	return {in_shape};
}
std::vector<aly::dim3> PowerLayer::getOutputDimensions() const {
	return {in_shape};
}
void PowerLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	const Tensor &x = *in_data[0];
	Tensor &y = *out_data[0];

	for (int i = 0; i < x.size(); i++) {
		std::transform(x[i].begin(), x[i].end(), y[i].begin(),
				[=](float x) {return scale * std::pow(x, factor);});
	}
}
void PowerLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	Tensor &dx = *in_grad[0];
	const Tensor &dy = *out_grad[0];
	const Tensor &x = *in_data[0];
	const Tensor &y = *out_data[0];
	for (int i = 0; i < x.size(); i++) {
		for (int j = 0; j < x[i].size(); j++) {
			// f(x) = (scale*x)^factor
			// ->
			//   dx = dy * df(x)
			//      = dy * scale * factor * (scale * x)^(factor - 1)
			//      = dy * scale * factor * (scale * x)^factor * (scale *
			//      x)^(-1)
			//      = dy * factor * y / x
			if (std::abs(x[i][j]) > 1e-10) {
				dx[i][j] = dy[i][j] * factor * y[i][j] / x[i][j];
			} else {
				dx[i][j] = dy[i][j] * scale * factor
						* std::pow(x[i][j], factor - 1.0f);
			}
		}
	}
}
}

