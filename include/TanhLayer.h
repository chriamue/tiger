/*
 * TanhLayer.h
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#ifndef INCLUDE_TANHLAYER_H_
#define INCLUDE_TANHLAYER_H_

#include "ActivationLayer.h"
namespace tgr {

class TanhLayer: public ActivationLayer {
public:
	TanhLayer(int in_width, int in_height, int in_channels) :
			ActivationLayer("tanh", in_width, in_height, in_channels) {
	}
	TanhLayer(int size) :
			ActivationLayer("tanh", size) {
	}
	virtual void forward_activation(const Storage &x, Storage &y) override;

	virtual void backward_activation(const Storage &x, const Storage &y,
			Storage &dx, const Storage &dy) override;

	virtual std::pair<float_t, float_t> scale() const override;
};
typedef std::shared_ptr<TanhLayer> TanhLayerPtr;
}

#endif /* INCLUDE_TANHLAYER_H_ */
