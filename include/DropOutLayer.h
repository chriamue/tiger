/*
 * DropOutLayer.h
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#ifndef INCLUDE_DROPOUTLAYER_H_
#define INCLUDE_DROPOUTLAYER_H_
#include "NeuralLayer.h"

namespace tgr {

/**
 * applies dropout to the input
 **/
class DropOutLayer: public NeuralLayer {
public:
	/**
	 * @param in_dim       [in] number of elements of the input
	 * @param dropout_rate [in] (0-1) fraction of the input units to be dropped
	 * @param phase        [in] initial state of the dropout
	 **/
	DropOutLayer(int in_dim, float dropout_rate, NetPhase phase =
			NetPhase::Train);
	virtual ~DropOutLayer() {
	}
	void setDropOutRate(float rate);
	float getDropOutRate() const;
	///< number of incoming connections for each output unit
	virtual int getFanInSize() const override;
	///< number of outgoing connections for each input unit
	virtual int getFanOutSize() const override;
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	/**
	 * set dropout-context (training-phase or test-phase)
	 **/
	virtual void setContext(const NetPhase& ctx) override;
	// currently used by tests only
	const std::vector<uint8_t> &getMask(int sample_index) const;
	std::vector<uint8_t> &getMask(int sample_index);
	void clearMask();
private:
	NetPhase phase;
	float dropout_rate;
	float scale;
	int in_size;
	std::vector<std::vector<uint8_t>> mask;
};

}

#endif /* INCLUDE_DROPOUTLAYER_H_ */
