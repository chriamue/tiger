/*
 * LinearLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "NeuralLayer.h"
namespace tgr {
class AddElementsLayer: public NeuralLayer {
public:
	AddElementsLayer(int num_args, int dim);
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
private:
	int num_args_;
	int dim_;
};
}
