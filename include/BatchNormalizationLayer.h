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
	void update_immidiately(bool update);
	void set_stddev(const Storage &stddev);
	void set_mean(const Storage &mean);
	void set_variance(const Storage &variance);
	float getEpsilon() const;
	float getMomentum() const;
private:
	void calc_stddev(const Storage &variance);

	void init();

	int in_channels_;
	int in_spatial_size_;

	tiny_dnn::net_phase phase_;
	float momentum_;
	float eps_;

	// mean/variance for this mini-batch
	Storage mean_current_;
	Storage variance_current_;

	Storage tmp_mean_;

	// moving average of mean/variance
	Storage mean_;
	Storage variance_;
	Storage stddev_;

	// for test
	bool update_immidiately_;
};
typedef std::shared_ptr<BatchNormalizationLayer> BatchNormalizationLayerPtr;

}

#endif /* INCLUDE_BATCHNORMALIZATIONLAYER_H_ */
