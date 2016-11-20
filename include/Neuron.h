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
		int layer;
		int bin;

		Terminal(int x=0, int y=0, int b=0, int l=-1) :x(x), y(y), bin(b), layer(l) {

		}
		bool operator ==(const Terminal & r) const {
			return (x == r.x && y == r.y && layer == r.layer && bin == r.bin);
		}
		bool operator !=(const Terminal & r) const {
			return (x != r.x || y != r.y || layer!=r.layer||bin!=r.bin);
		}
		bool operator <(const Terminal & r) const {
			return (std::make_tuple(x, y,layer,bin) < std::make_tuple(r.x, r.y, r.layer, r.bin));
		}
		bool operator >(const Terminal & r) const {
			return (std::make_tuple(x, y, layer, bin) < std::make_tuple(r.x, r.y, r.layer, r.bin));
		}
	};
	struct Signal {
		float value;
		float delta;
		Signal() :value(0.0f), delta(0.0f) {

		}
		Signal(float value) : value(value), delta(0.0f) {

		}
	};
	typedef std::shared_ptr<Signal> SignalPtr;
	class Neuron {
	protected:
		NeuronFunction transform;
		std::vector<SignalPtr> input;
		std::vector<SignalPtr> output;

	public:
		float value;
		float delta;
		friend class NeuralLayer;
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
		float& getOututWeight(size_t idx) {
			return output[idx]->value;
		}
		const float& getOutputWeight(size_t idx) const {
			return output[idx]->value;
		}
		Neuron(const NeuronFunction& func = ReLU(),float val=0.0f);
		void setFunction(const NeuronFunction& func) {
			transform = func;
		}
	};
}
#endif