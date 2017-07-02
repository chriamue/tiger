/*
 * ElementAddLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include <AddElementsLayer.h>
namespace tgr {
AddElementsLayer::AddElementsLayer(int num_args, int dim) :
		NeuralLayer("Add Elements",
				std::vector<ChannelType>(num_args, ChannelType::data), {
						ChannelType::data }), num_args_(num_args), dim_(dim) {
}
std::vector<aly::dim3> AddElementsLayer::getInputDimensions() const {
	return std::vector<aly::dim3>(num_args_, aly::dim3(dim_, 1, 1));
}
std::vector<aly::dim3> AddElementsLayer::getOutputDimensions() const {
	return {aly::dim3(dim_, 1, 1)};
}
void AddElementsLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	const Tensor &in1 = *in_data[0];
	Tensor &out = *out_data[0];

	out = in1;

	// @todo parallelize
	for (size_t sample = 0; sample < in1.size(); ++sample) {
		for (int i = 1; i < num_args_; i++) {
			std::transform((*in_data[i])[sample].begin(),
					(*in_data[i])[sample].end(), out[sample].begin(),
					out[sample].begin(),
					[](float_t x, float_t y) {return x + y;});
		}
	}
}
void AddElementsLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	CNN_UNREFERENCED_PARAMETER(in_data);
	CNN_UNREFERENCED_PARAMETER(out_data);
	for (int i = 0; i < num_args_; i++)
		*in_grad[i] = *out_grad[0];
}
}

