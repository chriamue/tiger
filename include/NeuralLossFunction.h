/*
 * NeuralLossFunction.h
 *
 *  Created on: Jul 6, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURALLOSSFUNCTION_H_
#define INCLUDE_NEURALLOSSFUNCTION_H_

#include "NeuralSignal.h"
namespace tgr {

class NeuralLossFunction {
public:
	struct Interface {
		virtual float f(const Storage &y, const Storage &t) const = 0;
		virtual Storage df(const Storage &y, const Storage &t) const = 0;
	};
private:
	template<class T> struct Impl: public Interface {
		T value;
		Impl(const T& value) :
				value(value) {
		}
		virtual float f(const Storage &y, const Storage &t) const override {
			return value.f(y, t);
		}
		virtual Storage df(const Storage &y, const Storage &t) const override {
			return value.df(y, t);
		}
	};
	std::shared_ptr<Interface> impl;
public:
	NeuralLossFunction() {
	}
	virtual ~NeuralLossFunction(){}
	NeuralLossFunction(const NeuralLossFunction& r) :
			impl(r.impl) {
	}
	template<class T> NeuralLossFunction(T* value) :
			impl(value) {
	}
	template<class T> NeuralLossFunction(const T & value) :
			impl(new Impl<T> { value }) {
	}
	NeuralLossFunction & operator =(const NeuralLossFunction & r) {
		impl = r.impl;
		return *this;
	}
	template<class T> NeuralLossFunction & operator =(const T & value) {
		return *this = NeuralLossFunction(value);
	}
	virtual float f(const Storage &y, const Storage &t) const {
		return impl->f(y, t);
	}
	virtual Storage df(const Storage &y, const Storage &t) const {
		return impl->df(y, t);
	}
	Storage gradient(const Storage& y, const Storage& t) const;
	std::vector<Storage> gradient(const std::vector<Storage> &y,
			const std::vector<Storage> &t) const;
	std::vector<Tensor> gradient(const std::vector<Tensor> &y,
			const std::vector<Tensor> &t,
			const std::vector<Tensor> &t_cost) const;
	Storage gradientLossFunction(const Storage &y, const Storage &t) const;
	std::vector<Storage> gradientLossFunction(const std::vector<Storage> &y,
			const std::vector<Storage> &t) const;
};
// mean-squared-error loss function for regression
class MSELossFunction: public NeuralLossFunction::Interface {
public:
	MSELossFunction() {
	}
	virtual float f(const Storage &y, const Storage &t) const override;
	virtual Storage df(const Storage &y, const Storage &t) const override;
};

// absolute loss function for regression
class AbsoluteLossFunction: public NeuralLossFunction::Interface {
public:
	AbsoluteLossFunction() {
	}
	virtual float f(const Storage &y, const Storage &t) const override;
	virtual Storage df(const Storage &y, const Storage &t) const override;
};
// absolute loss with epsilon range for regression
// epsilon range [-eps, eps] with eps = 1./fraction
class AbsoluteEpsLossFunction: public NeuralLossFunction::Interface {
protected:
	int fraction;
public:
	AbsoluteEpsLossFunction(int fraction = 8) :
			fraction(fraction) {
	}
	virtual float f(const Storage &y, const Storage &t) const override;
	virtual Storage df(const Storage &y, const Storage &t) const override;
};

// cross-entropy loss function for (multiple independent) binary classifications
class CrossEntropyLossFunction: public NeuralLossFunction::Interface {
public:
	CrossEntropyLossFunction() {
	}
	virtual float f(const Storage &y, const Storage &t) const override;
	virtual Storage df(const Storage &y, const Storage &t) const override;
};

// cross-entropy loss function for multi-class classification
class CrossEntropyMultiClassLossFunction: public NeuralLossFunction::Interface {
public:
	CrossEntropyMultiClassLossFunction() {
	}
	virtual float f(const Storage &y, const Storage &t) const override;
	virtual Storage df(const Storage &y, const Storage &t) const override;
};
void ApplyCostIfDefined(std::vector<Storage> &sample_gradient,
		const std::vector<Storage> &sample_cost);
}
#endif /* INCLUDE_NEURALLOSSFUNCTION_H_ */
