/*
 * NeuralLossFunction.h
 *
 *  Created on: Jul 6, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURALLOSSFUNCTION_H_
#define INCLUDE_NEURALLOSSFUNCTION_H_

#include "NeuralLayer.h"
namespace tgr {
class NeuralLossFunction {
	virtual float f(const Storage &y, const Storage &t) = 0;
	virtual Storage df(const Storage &y, const Storage &t) = 0;
};
// mean-squared-error loss function for regression
class MSELossFunction {
public:
	static float f(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		float d { 0.0 };

		for (int i = 0; i < y.size(); ++i)
			d += (y[i] - t[i]) * (y[i] - t[i]);

		return d / static_cast<float>(y.size());
	}

	static Storage df(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		Storage d(t.size());
		float factor = float(2) / static_cast<float>(t.size());

		for (int i = 0; i < y.size(); ++i)
			d[i] = factor * (y[i] - t[i]);

		return d;
	}
};

// absolute loss function for regression
class AbsoluteLossFunction {
public:
	static float f(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		float d { 0 };

		for (int i = 0; i < y.size(); ++i)
			d += std::abs(y[i] - t[i]);

		return d / static_cast<float>(y.size());
	}

	static Storage df(const Storage &y, const Storage &t) {
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
};

// absolute loss with epsilon range for regression
// epsilon range [-eps, eps] with eps = 1./fraction
template<int fraction>
class AbsoluteEpsLossFunction {
public:
	static float f(const Storage &y, const Storage &t) {
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

	static Storage df(const Storage &y, const Storage &t) {
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
};

// cross-entropy loss function for (multiple independent) binary classifications
class CrossEntropyLossFunction {
public:
	static float f(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		float d { 0 };

		for (int i = 0; i < y.size(); ++i)
			d += -t[i] * std::log(y[i])
					- (float(1) - t[i]) * std::log(float(1) - y[i]);

		return d;
	}

	static Storage df(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		Storage d(t.size());

		for (int i = 0; i < y.size(); ++i)
			d[i] = (y[i] - t[i]) / (y[i] * (float(1) - y[i]));

		return d;
	}
};

// cross-entropy loss function for multi-class classification
class CrossEntropyMultiClassLossFunction {
public:
	static float f(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		float d { 0.0 };

		for (int i = 0; i < y.size(); ++i)
			d += -t[i] * std::log(y[i]);

		return d;
	}

	static Storage df(const Storage &y, const Storage &t) {
		assert(y.size() == t.size());
		Storage d(t.size());

		for (int i = 0; i < y.size(); ++i)
			d[i] = -t[i] / y[i];

		return d;
	}
};

template<typename E>
Storage GradientLossFunction(const Storage &y, const Storage &t) {
	assert(y.size() == t.size());
	return E::df(y, t);
}

template<typename E>
std::vector<Storage> GradientLossFunction(const std::vector<Storage> &y,
		const std::vector<Storage> &t) {
	std::vector<Storage> grads(y.size());

	assert(y.size() == t.size());

	for (int i = 0; i < y.size(); i++)
		grads[i] = gradient<E>(y[i], t[i]);

	return grads;
}

inline void apply_cost_if_defined(std::vector<Storage> &sample_gradient,
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

// gradient for a minibatch
template<typename E>
std::vector<Tensor> gradient(const std::vector<Tensor> &y,
		const std::vector<Tensor> &t, const std::vector<Tensor> &t_cost) {
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

		gradients[sample] = gradient<E>(y[sample], t[sample]);

		if (sample < t_cost.size()) {
			apply_cost_if_defined(gradients[sample], t_cost[sample]);
		}
	}

	return gradients;
}
}
#endif /* INCLUDE_NEURALLOSSFUNCTION_H_ */
