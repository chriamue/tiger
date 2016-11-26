/*
* Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
#ifndef NEURONFUNCTION_H_
#define NEURONFUNCTION_H_
#include <AlloyMath.h>
namespace tgr {
	enum class NeuronFunctionType {
		Sigmoid, Tanh, ReLU, LeakyReLU, Linear, Constant
	};


	class NeuronFunction {
	public:
		struct Interface {
			virtual float forward(float t) const = 0;
			virtual float change(float t) const = 0;
			virtual NeuronFunctionType type() const = 0;
		};
	private:
		template<class T> struct Impl : public Interface {
			T value;
			Impl(const T& value) : value(value) {
			}
			virtual NeuronFunctionType type() const {
				return value.type();
			}
			virtual float forward(float t) const {
				return value.forward(t);
			}
			virtual float change(float t) const {
				return value.change(t);
			}

		};
		std::shared_ptr<Interface> impl;
	public:
		template<class Archive> void save(Archive& archive) const {
			if (impl.get()) {
				std::string typeName = MakeString() << impl->type();
				archive(cereal::make_nvp(typeName, impl->toString()));
			}
			else {
				archive("");
			}
		}
		template<class Archive> void load(Archive& archive) {

		}
		NeuronFunction() {
		}
		NeuronFunction clone() const;
		NeuronFunction(const NeuronFunction & r) :
			impl(r.impl) {
		}
		template<class T> NeuronFunction(const T & value) :
			impl(new Impl<T>{ value }) {
		}
		template<class T> NeuronFunction(T* value) :
			impl(value) {
		}
		NeuronFunction & operator =(const NeuronFunction & r) {
			impl = r.impl;
			return *this;
		}
		template<class T> NeuronFunction & operator =(const T & value) {
			return *this = NeuronFunction(value);
		}
		virtual NeuronFunctionType type() const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->type();
		}
		virtual float forward(float t) const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->forward(t);
		}
		virtual float change(float t) const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->change(t);
		}
		virtual inline ~NeuronFunction() {
		}
	};

	struct Sigmoid {
	public:
		Sigmoid() {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::Sigmoid;
		}
		float forward(float t) const {
			return 1.0f / (1 + std::exp(-t));
		}
		float change(float t) const {
			float f_t = forward(t);
			return f_t*(1 - f_t);
		}
		Sigmoid(const NeuronFunction& r) {
		}
		virtual ~Sigmoid() {
		}
	};
	struct Tanh {
	public:
		Tanh() {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::Tanh;
		}
		float forward(float t) const {
			float e = std::exp(2 * t);
			return (e - 1) / (e + 1);
		}
		float change(float t) const {
			float f_t = forward(t);
			return 1.0f - f_t*f_t;
		}
		Tanh(const NeuronFunction& r) {
		}
		virtual ~Tanh() {
		}
	};


	struct ReLU {
	public:
		ReLU() {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::ReLU;
		}
		float forward(float t) const {
			return std::max(0.0f, t);
		}
		float change(float t) const {
			float f_t = forward(t);
			return (f_t > 0.0f) ? 1.0f : 0.0f;
		}
		ReLU(const NeuronFunction& r) {
		}
		virtual ~ReLU() {
		}
	};
	struct Linear {
	public:
		Linear() {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::Linear;
		}
		float forward(float t) const {
			return t;
		}
		float change(float t) const {
			return 1.0f;
		}

		Linear(const NeuronFunction& r) {
		}
		virtual ~Linear() {
		}
	};
	struct Constant {
	protected:
		float* value;
	public:
		Constant() {
			value = nullptr;
		}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::Constant;
		}
		float forward(float t) const {
			return *value;
		}
		float change(float t) const {
			return 0.0f;
		}

		Constant(float* val):value(val) {
		}
		Constant(const NeuronFunction& func):value(static_cast<Constant>(func).value){
		}
		virtual ~Constant() {
		}
	};
	struct LeakyReLU {
	private:
		float eps;
	public:
		LeakyReLU(float eps = 0.01f) :eps(eps) {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::LeakyReLU;
		}
		float forward(float t) const {
			return (t > 0.0f) ? t : eps*t;
		}
		float change(float t) const {
			float f_t = forward(t);
			return (f_t > 0.0f) ? 1.0f : eps;
		}

		LeakyReLU(const NeuronFunction& func):eps(0.01f) {
		}
		virtual ~LeakyReLU() {
		}
	};

}
#endif