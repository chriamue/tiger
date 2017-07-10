/*
 * NeuralLossFunction.cpp
 *
 *  Created on: Jul 8, 2017
 *      Author: blake
 */

#include "NeuralLossFunction.h"
namespace tgr {
Storage NeuralLossFunction::gradientLossFunction(const Storage &y,
		const Storage &t) const {
	assert(y.size() == t.size());
	return this->df(y, t);
}

Storage NeuralLossFunction::gradient(const Storage& y, const Storage& t) const {
	assert(y.size() == t.size());
	return df(y, t);
}
std::vector<Storage> NeuralLossFunction::gradient(const std::vector<Storage> &y,
		const std::vector<Storage> &t) const {
	std::vector<Storage> grads(y.size());
	assert(y.size() == t.size());
	for (int i = 0; i < (int) y.size(); i++)
		grads[i] = gradient(y[i], t[i]);
	return grads;
}
std::vector<Tensor> NeuralLossFunction::gradient(const std::vector<Tensor>& y,
		const std::vector<Tensor> &t, const std::vector<Tensor> &t_cost) const {
	const int sample_count = static_cast<int>(y.size());
	const int channel_count = static_cast<int>(y[0].size());
	std::vector<Tensor> gradients(sample_count);
	CNN_UNREFERENCED_PARAMETER(channel_count);
	assert(y.size() == t.size());
	assert(t_cost.empty() || t_cost.size() == t.size());
	// @todo add parallelism
	for (int sample = 0; sample < sample_count; ++sample) {
		assert(y[sample].size() == channel_count);
		assert(t[sample].size() == channel_count);
		assert(
				t_cost.empty() || t_cost[sample].empty()
						|| t_cost[sample].size() == channel_count);
		gradients[sample] = this->gradient(y[sample], t[sample]);
		if (sample < t_cost.size()) {
			ApplyCostIfDefined(gradients[sample], t_cost[sample]);
		}
	}
	return gradients;
}
std::vector<Storage> NeuralLossFunction::gradientLossFunction(
		const std::vector<Storage> &y, const std::vector<Storage> &t) const {
	std::vector<Storage> grads(y.size());
	assert(y.size() == t.size());
	for (int i = 0; i < y.size(); i++)
		grads[i] = gradient(y[i], t[i]);
	return grads;
}
float MSELossFunction::f(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	float d { 0.0 };

	for (int i = 0; i < y.size(); ++i)
		d += (y[i] - t[i]) * (y[i] - t[i]);

	return d / static_cast<float>(y.size());
}

Storage MSELossFunction::df(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	Storage d(t.size());
	float factor = float(2) / static_cast<float>(t.size());

	for (int i = 0; i < y.size(); ++i)
		d[i] = factor * (y[i] - t[i]);
	return d;
}
float AbsoluteLossFunction::f(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	float d { 0 };

	for (int i = 0; i < y.size(); ++i)
		d += std::abs(y[i] - t[i]);

	return d / static_cast<float>(y.size());
}
Storage AbsoluteLossFunction::df(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	Storage d(t.size());
	float factor = float(1) / static_cast<float>(t.size());

	for (int i = 0; i < y.size(); ++i) {
		float sign = y[i] - t[i];
		if (sign < float { 0.f })
			d[i] = -factor;
		else if (sign > float { 0.f })
			d[i] = factor;
		else
			d[i] = {0};
		}

	return d;
}
float AbsoluteEpsLossFunction::f(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	float d { 0 };
	const float eps = float(1) / fraction;

	for (int i = 0; i < y.size(); ++i) {
		float diff = std::abs(y[i] - t[i]);
		if (diff > eps)
			d += diff;
	}
	return d / static_cast<float>(y.size());
}
Storage AbsoluteEpsLossFunction::df(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	Storage d(t.size());
	const float factor = float(1) / static_cast<float>(t.size());
	const float eps = float(1) / fraction;

	for (int i = 0; i < y.size(); ++i) {
		float sign = y[i] - t[i];
		if (sign < -eps)
			d[i] = -factor;
		else if (sign > eps)
			d[i] = factor;
		else
			d[i] = 0.f;
	}
	return d;
}
float CrossEntropyLossFunction::f(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	float d { 0 };

	for (int i = 0; i < y.size(); ++i)
		d += -t[i] * std::log(y[i])
				- (float(1) - t[i]) * std::log(float(1) - y[i]);

	return d;
}
Storage CrossEntropyLossFunction::df(const Storage &y, const Storage &t) const {
	assert(y.size() == t.size());
	Storage d(t.size());

	for (int i = 0; i < y.size(); ++i)
		d[i] = (y[i] - t[i]) / (y[i] * (float(1) - y[i]));

	return d;
}
float CrossEntropyMultiClassLossFunction::f(const Storage &y,
		const Storage &t) const {
	assert(y.size() == t.size());
	float d { 0.0 };

	for (int i = 0; i < y.size(); ++i)
		d += -t[i] * std::log(y[i]);

	return d;
}
Storage CrossEntropyMultiClassLossFunction::df(const Storage &y,
		const Storage &t) const {
	assert(y.size() == t.size());
	Storage d(t.size());

	for (int i = 0; i < y.size(); ++i)
		d[i] = -t[i] / y[i];

	return d;
}
void ApplyCostIfDefined(std::vector<Storage> &sample_gradient,
		const std::vector<Storage> &sample_cost) {
	if (sample_gradient.size() == sample_cost.size()) {
		// @todo consider adding parallelism
		const int channel_count = static_cast<int>(sample_gradient.size());
		for (size_t channel = 0; channel < channel_count; ++channel) {
			if (sample_gradient[channel].size()
					== sample_cost[channel].size()) {
				const size_t element_count = sample_gradient[channel].size();
				// @todo optimize? (use AVX or so)
				for (size_t element = 0; element < element_count; ++element) {
					sample_gradient[channel][element] *=
							sample_cost[channel][element];
				}
			}
		}
	}
}

}

