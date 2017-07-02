/*
 * InputLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "InputLayer.h"
namespace tgr {
InputLayer::InputLayer(const aly::dim3& shape) :
		NeuralLayer("Input", { ChannelType::data }, { ChannelType::data }), shape_(
				shape) {
}
InputLayer::InputLayer(int in_dim) :
		NeuralLayer("Input", { ChannelType::data }, { ChannelType::data }), shape_(
				aly::dim3(in_dim, 1, 1)) {
}
std::vector<aly::dim3> InputLayer::getInputDimensions() const {
	return {shape_};
}
std::vector<aly::dim3> InputLayer::getOutputDimensions() const {
	return {shape_};
}
void InputLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	*out_data[0] = *in_data[0];
}
void InputLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {

}
}

