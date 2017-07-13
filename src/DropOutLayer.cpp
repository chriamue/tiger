/*
 * DropOutLayer.cpp
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#include "DropOutLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
namespace tgr {
DropOutLayer::DropOutLayer(int in_dim, float dropout_rate, NetPhase phase) :
		NeuralLayer("Drop Out", { ChannelType::data }, { ChannelType::data }), phase(
				phase), dropout_rate(dropout_rate), scale(
				float(1) / (float(1) - dropout_rate)), in_size(in_dim) {
	mask.resize(1, std::vector<uint8_t>(in_dim));
	clearMask();
}
void DropOutLayer::setDropOutRate(float rate) {
	dropout_rate = rate;
	scale = float(1) / (float(1) - dropout_rate);
}
float DropOutLayer::getDropOutRate() const {
	return dropout_rate;
}
///< number of incoming connections for each output unit
int DropOutLayer::getFanInSize() const {
	return 1;
}
///< number of outgoing connections for each input unit
int DropOutLayer::getFanOutSize() const {
	return 1;
}
std::vector<aly::dim3> DropOutLayer::getInputDimensions() const {
	return {aly::dim3(in_size, 1, 1)};
}

std::vector<aly::dim3> DropOutLayer::getOutputDimensions() const {
	return {aly::dim3(in_size, 1, 1)};
}

/**
 * set dropout-context (training-phase or test-phase)
 **/
void DropOutLayer::setContext(const NetPhase& ctx) {
	phase = ctx;
}
// currently used by tests only
const std::vector<uint8_t> &DropOutLayer::getMask(int sample_index) const {
	return mask[sample_index];
}
std::vector<uint8_t> &DropOutLayer::getMask(int sample_index) {
	return mask[sample_index];
}
void DropOutLayer::clearMask() {
	for (auto &sample : mask) {
		std::fill(sample.begin(), sample.end(), 0);
	}
}
void DropOutLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	Tensor &prev_delta = *in_grad[0];
	const Tensor &curr_delta = *out_grad[0];
	CNN_UNREFERENCED_PARAMETER(in_data);
	CNN_UNREFERENCED_PARAMETER(out_data);
	for_i(prev_delta.size(), [&](size_t sample) {
		// assert(prev_delta[sample].size() == curr_delta[sample].size());
		// assert(mask_[sample].size() == prev_delta[sample].size());
			size_t sz = prev_delta[sample].size();
			for (size_t i = 0; i < sz; ++i) {
				prev_delta[sample][i] = mask[sample][i] * curr_delta[sample][i];
			}
		});
}

void DropOutLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	const Tensor &in = *in_data[0];
	Tensor &out = *out_data[0];

	const size_t sample_count = in.size();

	if (mask.size() < sample_count) {
		mask.resize(sample_count, mask[0]);
	}

	for_i(sample_count, [&](size_t sample) {
		std::vector<uint8_t> &mask = this->mask[sample];
		const Storage &in_vec = in[sample];
		Storage &out_vec = out[sample];
		if (phase == NetPhase::Train) {
			for (size_t i = 0; i < in_vec.size(); i++)
			mask[i] = bernoulli(dropout_rate);

			for (size_t i = 0; i < in_vec.size(); i++)
			out_vec[i] = mask[i] * scale * in_vec[i];
		} else {
			for (size_t i = 0, end = in_vec.size(); i < end; i++)
			out_vec[i] = in_vec[i];
		}
	});
}

}

