/*
 * ConcatLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_CONCATLAYER_H_
#define INCLUDE_CONCATLAYER_H_
#include "NeuralLayer.h"
namespace tgr {
class ConcatLayer: public NeuralLayer {
public:
	ConcatLayer(const std::vector<aly::dim3> &in_shapes);
	ConcatLayer(int num_args, int ndim);
	void set_outshape();
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	virtual void getStencilInput(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override;
	virtual void getStencilWeight(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override {
		stencil.clear();
	}
	virtual bool getStencilBias(const aly::int3& pos, aly::int3& stencil) const
			override {
		return false;
	}
private:
	std::vector<aly::dim3> in_shapes;
	aly::dim3 out_shape;
};
typedef std::shared_ptr<ConcatLayer> ConcatLayerPtr;
}

#endif /* INCLUDE_CONCATLAYER_H_ */
