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
#include <memory>
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
		std::vector<Neuron*> input;
		std::vector<Neuron*> output;
		Signal() :value(0.0f) {

		}
		Signal(float value) : value(value) {

		}
		void addInput(Neuron* n) {
			input.push_back(n);
		}
		void addOutput(Neuron* n) {
			output.push_back(n);
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
		bool active;
		friend class NeuralLayer;
		size_t getInputWeightSize() const {
			return input.size();
		}
		size_t getOutputWeightSize() const {
			return output.size();
		}
		size_t getInputNeuronSize() const {
			size_t count = 0;
			for (auto in : input) {
				count += in->input.size();
			}
			return count;
		}
		size_t getOutputNeuronSize() const {
			size_t count = 0;
			for (auto out : output) {
				count += out->output.size();
			}
			return count;
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
		std::vector<Neuron*> getInputNeurons() const;
		void getInputNeurons(std::vector<Neuron*>& out) const;
		std::vector<Neuron*> getOutputNeurons() const;
		void getOutputNeurons(std::vector<Neuron*>& out) const;
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
		}
		void addOutput(const SignalPtr& s) {
			output.push_back(s);
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
			Bias() :Neuron(Constant(&value), false, 1.0f) {

			}
	};
	void MakeConnection(Neuron* src, const std::shared_ptr<Signal>& signal,Neuron* dest);
}
#endif