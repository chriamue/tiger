/*
 * ActivationLayer.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "ActivationLayer.h"
#include "tiny_dnn/tiny_dnn.h"
namespace tgr {
void ActivationLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	const Tensor &x = *in_data[0];
	Tensor &y = *out_data[0];
	tiny_dnn::for_i(x.size(), [&](int i) {forward_activation(x[i], y[i]);});
}
void ActivationLayer::backwardPropagation(const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data, std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	Tensor&dx = *in_grad[0];
	const Tensor&dy = *out_grad[0];
	const Tensor&x = *in_data[0];
	const Tensor&y = *out_data[0];
	tiny_dnn::for_i(x.size(),
			[&](size_t i) {backward_activation(x[i], y[i], dx[i], dy[i]);});
}
}
