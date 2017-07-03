/*
 * BatchNormalizationLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "BatchNormalizationLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
namespace tgr {

/**
 * @param prev_layer      [in] previous layer to be connected with this layer
 * @param epsilon         [in] small positive value to avoid zero-division
 * @param momentum        [in] momentum in the computation of the exponential
 *average of the mean/stddev of the data
 * @param phase           [in] specify the current context (train/test)
 **/
BatchNormalizationLayer::BatchNormalizationLayer(const NeuralLayer &prev_layer,
		float epsilon, float momentum, tiny_dnn::net_phase phase) :
		NeuralLayer("Batch Normalization", { ChannelType::data }, {
				ChannelType::data }), in_channels(prev_layer.getOutputDimensions()[0].z), in_spatial_size(prev_layer.getOutputDimensions()[0].x
						* prev_layer.getOutputDimensions()[0].y), phase(phase), momentum(
				momentum), eps(epsilon), update_immidiately(false) {
	init();
}

///< number of incoming connections for each output unit
int BatchNormalizationLayer::getFanInSize() const {
	return 1;
}

///< number of outgoing connections for each input unit
int BatchNormalizationLayer::getFanOutSize() const {
	return 1;
}

std::vector<aly::dim3> BatchNormalizationLayer::getInputDimensions() const {
	return {aly::dim3(in_spatial_size, 1, in_channels)};
}

std::vector<aly::dim3> BatchNormalizationLayer::getOutputDimensions() const {
	return {aly::dim3(in_spatial_size, 1, in_channels)};
}
float BatchNormalizationLayer::getEpsilon() const {
	return eps;
}

float BatchNormalizationLayer::getMomentum() const {
	return momentum;
}
void BatchNormalizationLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data,
		std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	Tensor &prev_delta = *in_grad[0];
	Tensor &curr_delta = *out_grad[0];
	const Tensor &curr_out = *out_data[0];
	int num_samples = static_cast<int>(curr_out.size());

	CNN_UNREFERENCED_PARAMETER(in_data);

	Tensor delta_dot_y = curr_out;
	Storage mean_delta_dot_y, mean_delta, mean_Y;

	for (int i = 0; i < num_samples; i++) {
		for (int j = 0; j < curr_out[0].size(); j++) {
			delta_dot_y[i][j] *= curr_delta[i][j];
		}
	}
	moments(delta_dot_y, in_spatial_size, in_channels, mean_delta_dot_y);
	moments(curr_delta, in_spatial_size, in_channels, mean_delta);
// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
//
// dE(Y)/dX =
//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
//     ./ sqrt(var(X) + eps)
//
	tiny_dnn::for_i(num_samples, [&](int i) {
		for (int j = 0; j < in_channels; j++) {
			for (int k = 0; k < in_spatial_size; k++) {
				int index = j * in_spatial_size + k;

				prev_delta[i][index] = curr_delta[i][index] - mean_delta[j] -
				mean_delta_dot_y[j] * curr_out[i][index];

				// stddev_ is calculated in the forward pass
			prev_delta[i][index] /= stddevStorage[j];
		}
	}
});
}

void BatchNormalizationLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	Storage &mean = (phase == net_phase::train) ? mean_current : meanStorage;
	Storage &variance =
			(phase == net_phase::train) ? variance_current : varianceStorage;
	Tensor &in = *in_data[0];
	Tensor &out = *out_data[0];

	if (phase == net_phase::train) {
		// calculate mean/variance from this batch in train phase
		moments(*in_data[0], in_spatial_size, in_channels, mean, variance);
	}

// y = (x - mean) ./ sqrt(variance + eps)
	calc_stddev(variance);

	for_i(in_data[0]->size(), [&](int i) {
		const float_t *inptr = &in[i][0];
		float_t *outptr = &out[i][0];

		for (size_t j = 0; j < in_channels; j++) {
			float_t m = mean[j];

			for (size_t k = 0; k < in_spatial_size; k++) {
				*outptr++ = (*inptr++ - m) / stddevStorage[j];
			}
		}
	});

	if (phase == tiny_dnn::net_phase::train && update_immidiately) {
		meanStorage = mean_current;
		varianceStorage = variance_current;
	}
}

void BatchNormalizationLayer::setContext(tiny_dnn::net_phase ctx) {
	phase = ctx;
}
void BatchNormalizationLayer::post() {
	for (int i = 0; i < meanStorage.size(); i++) {
		meanStorage[i] = momentum * meanStorage[i] + (1 - momentum) * mean_current[i];
		varianceStorage[i] = momentum * varianceStorage[i]
				+ (1 - momentum) * variance_current[i];
	}
}

void BatchNormalizationLayer::updateImmidiately(bool update) {
	update_immidiately = update;
}

void BatchNormalizationLayer::setStddev(const Storage &stddev) {
	stddevStorage = stddev;
}

void BatchNormalizationLayer::setMean(const Storage &mean) {
	meanStorage = mean;
}

void BatchNormalizationLayer::setVariance(const Storage &variance) {
	varianceStorage = variance;
	calc_stddev(variance);
}



void BatchNormalizationLayer::calc_stddev(const Storage &variance) {
	for (size_t i = 0; i < in_channels; i++) {
		stddevStorage[i] = sqrt(variance[i] + eps);
	}
}

void BatchNormalizationLayer::init() {
	mean_current.resize(in_channels);
	meanStorage.resize(in_channels);
	variance_current.resize(in_channels);
	varianceStorage.resize(in_channels);
	tmp_mean.resize(in_channels);
	stddevStorage.resize(in_channels);
}
}

