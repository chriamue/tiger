/*
 * NeuralOptimizer.h
 *
 *  Created on: Jul 6, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURALOPTIMIZER_H_
#define INCLUDE_NEURALOPTIMIZER_H_

#include "NeuralLayer.h"
namespace tgr {
/**
 * base class of optimizer
 * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss
 *function)
 **/

class NeuralOptimizer {
public:
	struct Interface {
		virtual void update(const Storage &dW, Storage &W,
				bool parallelize) = 0;
		virtual void reset() = 0;
	};
private:
	template<class T> struct Impl: public Interface {
		T value;
		virtual void update(const Storage &dW, Storage &W, bool parallelize)
				override {
			value.update(dW, W, parallelize);
		}
		virtual void reset() override {
			value.reset();
		}
	};
	std::shared_ptr<Interface> impl;
public:
	NeuralOptimizer() = default;
	NeuralOptimizer(const NeuralOptimizer &) = default;
	NeuralOptimizer(NeuralOptimizer &&) = default;
	NeuralOptimizer &operator=(const NeuralOptimizer &) = default;
	NeuralOptimizer &operator=(NeuralOptimizer &&) = default;
	virtual ~NeuralOptimizer() = default;
	virtual void update(const Storage &dW, Storage &W, bool parallelize) {
		impl->update(dW, W, parallelize);
	}
	virtual void reset() {
		impl->reset();
	}
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct AdagradOptimizer: public NeuralOptimizer::Interface {
	AdagradOptimizer();
	void update(const Storage &dW, Storage &W, bool parallelize) override;
	float_t alpha;  // learning rate
	void reset() override {
		for (auto &e : E_)
			e.clear();
	}
private:
	float_t eps;
	static const int N = 1;
protected:
	template<int Index>
	Storage &get(const Storage &key) {
		static_assert(Index < N, "index out of range");
		if (E_[Index][&key].empty())
			E_[Index][&key].resize(key.size(), float_t());
		return E_[Index][&key];
	}
	std::unordered_map<const Storage *, Storage> E_[N];
};
/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSpropOptimizer: public NeuralOptimizer::Interface {
	RMSpropOptimizer();
	void update(const Storage &dW, Storage &W, bool parallelize) override;
	void reset() override {
		for (auto &e : E_)
			e.clear();
	}
	float_t alpha;  // learning rate
	float_t mu;     // decay term
private:
	float_t eps;  // constant value to avoid zero-division
	static const int N = 1;
protected:
	template<int Index>
	Storage &get(const Storage &key) {
		static_assert(Index < N, "index out of range");
		if (E_[Index][&key].empty())
			E_[Index][&key].resize(key.size(), float_t());
		return E_[Index][&key];
	}
	std::unordered_map<const Storage *, Storage> E_[N];
};

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct AdamOptimizer: public NeuralOptimizer::Interface {
	AdamOptimizer();
	void update(const Storage &dW, Storage &W, bool parallelize) override;
	void reset() override {
		for (auto &e : E_)
			e.clear();
	}
	float_t alpha;  // learning rate
	float_t b1;     // decay term
	float_t b2;     // decay term
	float_t b1_t;   // decay term power t
	float_t b2_t;   // decay term power t

private:
	float_t eps;  // constant value to avoid zero-division
	static const int N = 1;
protected:
	template<int Index>
	Storage &get(const Storage &key) {
		static_assert(Index < N, "index out of range");
		if (E_[Index][&key].empty())
			E_[Index][&key].resize(key.size(), float_t());
		return E_[Index][&key];
	}
	std::unordered_map<const Storage *, Storage> E_[N];
};

/**
 * SGD without momentum
 *
 * slightly faster than tiny_dnn::momentum
 **/
struct GradientDescentOptimizer: public NeuralOptimizer::Interface {
	GradientDescentOptimizer();
	virtual void update(const Storage &dW, Storage &W, bool parallelize)
			override;
	virtual void reset() override {
	}
	float_t alpha;   // learning rate
	float_t lambda;  // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct MomentumOptimizer: public NeuralOptimizer::Interface {
public:
	MomentumOptimizer();
	virtual void update(const Storage &dW, Storage &W, bool parallelize)
			override;
	virtual void reset() override {
		for (auto &e : E_)
			e.clear();
	}
	float_t alpha;   // learning rate
	float_t lambda;  // weight decay
	float_t mu;      // momentum
protected:
	static const int N = 1;
	template<int Index>
	Storage &get(const Storage &key) {
		static_assert(Index < N, "index out of range");
		if (E_[Index][&key].empty())
			E_[Index][&key].resize(key.size(), float_t());
		return E_[Index][&key];
	}
	std::unordered_map<const Storage *, Storage> E_[N];
};
}

#endif /* INCLUDE_NEURALOPTIMIZER_H_ */
