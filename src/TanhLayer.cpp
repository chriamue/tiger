/*
 * TanhLayer.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "TanhLayer.h"
namespace tgr {

void TanhLayer::forward_activation(const Storage &x, Storage &y) {
	for (int j = 0; j < x.size(); j++) {
		y[j] = std::tanh(x[j]);
	}
}

void TanhLayer::backward_activation(const Storage &x, const Storage &y,
		Storage &dx, const Storage &dy) {
	for (int j = 0; j < x.size(); j++) {
		// dx = dy * (gradient of tanh)
		dx[j] = dy[j] * (1.0f - y[j] * y[j]);
	}
}

std::pair<float_t, float_t> TanhLayer::scale() const {
	return std::make_pair(float_t(-0.8), float_t(0.8));
}
}

