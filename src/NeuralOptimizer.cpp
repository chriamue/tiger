/*
 * NeuralOptimizer.h
 *
 *  Created on: Jul 6, 2017
 *      Author: blake
 */
#include "tiny_dnn/tiny_dnn.h"
#include "NeuralOptimizer.h"
namespace tgr {
AdagradOptimizer::AdagradOptimizer() :
		alpha(float_t(0.01)), eps(float_t(1e-8)) {
}
void AdagradOptimizer::update(const Storage &dW, Storage &W, bool parallelize) {
	Storage &g = get<0>(W);
	tiny_dnn::for_i(parallelize, static_cast<int>(W.size()), [&](int i) {
		g[i] += dW[i] * dW[i];
		W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
	});
}
/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
RMSpropOptimizer::RMSpropOptimizer() :
		alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {
}
void RMSpropOptimizer::update(const Storage &dW, Storage &W, bool parallelize) {
	Storage &g = get<0>(W);

	tiny_dnn::for_i(parallelize, static_cast<int>(W.size()), [&](int i) {
		g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
		W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
	});
}

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
AdamOptimizer::AdamOptimizer() :
		alpha(float_t(0.001)), b1(float_t(0.9)), b2(float_t(0.999)), b1_t(
				float_t(0.9)), b2_t(float_t(0.999)), eps(float_t(1e-8)) {
}
void AdamOptimizer::update(const Storage &dW, Storage &W, bool parallelize) {
	Storage &mt = get<0>(W);
	Storage &vt = get<1>(W);

	b1_t *= b1;
	b2_t *= b2;

	tiny_dnn::for_i(parallelize, static_cast<int>(W.size()), [&](int i) {
		mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
		vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

		W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
		std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
	});
}

/**
 * SGD without momentum
 *
 * slightly faster than tiny_dnn::momentum
 **/
GradientDescentOptimizer::GradientDescentOptimizer() :
		alpha(float_t(0.01)), lambda(float_t(0)) {
}
void GradientDescentOptimizer::update(const Storage &dW, Storage &W,
		bool parallelize) {
	tiny_dnn::for_i(parallelize, static_cast<int>(W.size()),
			[&](int i) {W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);});
}

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
MomentumOptimizer::MomentumOptimizer() :
		alpha(float_t(0.01)), lambda(float_t { 0 }), mu(float_t(0.9)) {
}
void MomentumOptimizer::reset() {
	for (auto &e : E_)
		e.clear();
}
void MomentumOptimizer::update(const Storage &dW, Storage &W,
		bool parallelize) {
	Storage &dWprev = get<0>(W);
	tiny_dnn::for_i(parallelize, static_cast<int>(W.size()), [&](int i) {
		float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
		W[i] += V;
		dWprev[i] = V;
	});
}
}
