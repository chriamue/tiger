/*
 * BatchNormalizationLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_BATCHNORMALIZATIONLAYER_H_
#define INCLUDE_BATCHNORMALIZATIONLAYER_H_
#include "NeuralLayer.h"
namespace tgr {
class BatchNormalizationLayer: public NeuralLayer {
public:

	/**
	 * @param prev_layer      [in] previous layer to be connected with this layer
	 * @param epsilon         [in] small positive value to avoid zero-division
	 * @param momentum        [in] momentum in the computation of the exponential
	 *average of the mean/stddev of the data
	 * @param phase           [in] specify the current context (train/test)
	 **/
	BatchNormalizationLayer(const NeuralLayer &prev_layer, float epsilon = 1e-5,
			float momentum = 0.999, tiny_dnn::net_phase phase =
					tiny_dnn::net_phase::train);
	///< number of incoming connections for each output unit
	virtual int getFanInSize() const override;
	///< number of outgoing connections for each input unit
	virtual int getFanOutSize() const override;
	std::vector<aly::dim3> getInputDimensions() const override;
	std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	void setContext(tiny_dnn::net_phase ctx);
	virtual void post() override;
	void updateImmidiately(bool update);
	void setStddev(const Storage &stddev);
	void setMean(const Storage &mean);
	void setVariance(const Storage &variance);
	float getEpsilon() const;
	float getMomentum() const;
	virtual void getStencilInput(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override {
		stencil = std::vector<aly::int3> { pos };
	}
	virtual void getStencilWeight(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override {
		stencil.clear();
	}
	virtual bool getStencilBias(const aly::int3& pos, aly::int3& stencil) const
			override {
		return false;
	}
private:
	void calc_stddev(const Storage &variance);

	void init();

	int in_channels;
	int in_spatial_size;

	tiny_dnn::net_phase phase;
	float momentum;
	float eps;

	// mean/variance for this mini-batch
	Storage mean_current;
	Storage variance_current;

	Storage tmp_mean;

	// moving average of mean/variance
	Storage meanStorage;
	Storage varianceStorage;
	Storage stddevStorage;

	// for test
	bool update_immidiately;
};
typedef std::shared_ptr<BatchNormalizationLayer> BatchNormalizationLayerPtr;

}

#endif /* INCLUDE_BATCHNORMALIZATIONLAYER_H_ */
