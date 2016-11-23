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
#ifndef NEURON_H_
#define NEURON_H_
#include <AlloyMath.h>
#include "NeuronFunction.h"

namespace tgr {
	class NeuralLayer;
	struct Terminal {
		int x;
		int y;
		NeuralLayer* layer;
		Terminal(int x=0, int y=0,  NeuralLayer* l=nullptr) :x(x), y(y), layer(l) {

		}
		Terminal(int x = 0, int y = 0,  const std::shared_ptr<NeuralLayer>& l = nullptr) :x(x), y(y),  layer(l.get()) {

		}
		bool operator ==(const Terminal & r) const;
		bool operator !=(const Terminal & r) const;
		bool operator <(const Terminal & r) const;
		bool operator >(const Terminal & r) const;
	};
	class Neuron;
	struct Signal {
		float value;
		float delta;
		std::vector<Neuron*> input;
		std::vector<Neuron*> output;
		Signal() :value(0.0f), delta(0.0f) {

		}
		Signal(float value) : value(value), delta(0.0f) {

		}
	};
	typedef std::shared_ptr<Signal> SignalPtr;
	class Neuron {
	protected:
		NeuronFunction transform;
		std::shared_ptr<Neuron> bias;
		std::vector<SignalPtr> input;
		std::vector<SignalPtr> output;

	public:
		float value;
		float delta;
		friend class NeuralLayer;
		size_t getInputSize() const {
			return input.size();
		}
		size_t getOutputSize() const {
			return output.size();
		}
		float evaluate();
		const SignalPtr& getInput(size_t idx) const {
			return input[idx];
		}
		const SignalPtr& getOutput(size_t idx) const {
			return output[idx];
		}
		SignalPtr& getInput(size_t idx) {
			return input[idx];
		}
		SignalPtr& getOutput(size_t idx) {
			return output[idx];
		}
		bool hasBias() const {
			return (bias.get() != nullptr);
		}
		SignalPtr getBiasSignal() const {
			if (hasBias()) {
				return bias->output.front();
			}
			else {
				return SignalPtr();
			}
		}
		float getBias() const {
			if (hasBias()) {
				return bias->output.front()->value;
			}
			else {
				return 0;
			}
		}
		void addInput(const SignalPtr& s) {
			input.push_back(s);
			s->output.push_back(this);
		}
		void addOutput(const SignalPtr& s) {
			output.push_back(s);
			s->input.push_back(this);
		}
		float& getInputWeight(size_t idx) {
			return input[idx]->value;
		}
		const float& getInputWeight(size_t idx) const {
			return input[idx]->value;
		}
		float& getOutputWeight(size_t idx) {
			return output[idx]->value;
		}
		const float& getOutputWeight(size_t idx) const {
			return output[idx]->value;
		}
		Neuron(const NeuronFunction& func = ReLU(),bool bias=false,float val=0.0f);
		void setFunction(const NeuronFunction& func) {
			transform = func;
		}
	};
	class Bias : public Neuron {
		public:		
			Bias() :Neuron(Constant(), false, 1.0f) {

			}
	};
}
#endif