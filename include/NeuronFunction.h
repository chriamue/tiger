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
#ifndef NeuronFunction_H_
#define NeuronFunction_H_
#include <AlloyMath.h>
namespace jc {
	enum class NeuronFunctionType {
		Sigmoid, Tanh, ReLU, LeakyReLU
	};


	class NeuronFunction {
	public:
		struct Interface {
			virtual float forward(float t) const = 0;
			virtual float forwardChange(float t) const = 0;
			virtual float backward(float t) const = 0;
			virtual float backwardChange(float t) const = 0;
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
			virtual float forwardChange(float t) const {
				return value.forwardChange(t);
			}
			virtual float backward(float t) const {
				return value.backward(t);
			}
			virtual float backwardChange(float t) const {
				return value.backwardChange(t);
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
		virtual float forwardChange(float t) const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->forwardChange(t);
		}
		virtual float backward(float t) const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->backward(t);
		}
		virtual float backwardChange(float t) const {
			if (impl.get() == nullptr)
				throw std::runtime_error("NeuronFunction type has not been defined.");
			return impl->backwardChange(t);
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
		float forwardChange(float t) const {
			float f_t = forward(t);
			return f_t*(1 - f_t);
		}
		float backward(float f) const {
			return -std::log(1.0f / f - 1.0f);
		}
		float backwardChange(float f) const {
			return 1.0f / (f*f*(1.0f / f - 1.0f));
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
		float forwardChange(float t) const {
			float f_t = forward(t);
			return 1.0f - f_t*f_t;
		}
		float backward(float t) const {
			return 0.5f*std::log((1 + t) / (1 - t));
		}
		float backwardChange(float t) const {
			return 1.0f / (1.0f - t*t);
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
		float forwardChange(float t) const {
			float f_t = forward(t);
			return (f_t > 0.0f) ? 1.0f : 0.0f;
		}
		float backward(float t) const {
			return std::max(0.0f, t);
		}
		float backwardChange(float t) const {
			return (t > 0.0f) ? 1.0f : 0.0f;
		}
		ReLU(const NeuronFunction& r) {
		}
		virtual ~ReLU() {
		}
	};
	struct LeakyReLU {
	private:
		const float eps;
	public:
		LeakyReLU(float eps = 0.01f) :eps(eps) {}
		NeuronFunctionType virtual type() const {
			return NeuronFunctionType::LeakyReLU;
		}
		float forward(float t) const {
			return (t > 0.0f) ? t : eps*t;
		}
		float forwardChange(float t) const {
			float f_t = forward(t);
			return (f_t > 0.0f) ? 1.0f : eps;
		}
		float backward(float t) const {
			return (t > 0.0f) ? t : t / eps;
		}
		float backwardChange(float t) const {
			return (t > 0.0f) ? 1.0f : 1.0f / eps;
		}
		LeakyReLU(const NeuronFunction& r, float eps = 0.01f) :eps(eps) {
		}
		virtual ~LeakyReLU() {
		}
	};

}
#endif