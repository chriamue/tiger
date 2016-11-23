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
#include "Neuron.h"
#include "NeuralLayer.h"
namespace tgr {
	bool Terminal::operator ==(const Terminal & r) const {
		return (x == r.x && y == r.y && layer == r.layer);
	}
	bool Terminal::operator !=(const Terminal & r) const {
		return (x != r.x || y != r.y || layer != r.layer);
	}
	bool Terminal::operator <(const Terminal & r) const {
		return (std::make_tuple(x, y, (layer) ? layer->id : -1) < std::make_tuple(r.x, r.y, (layer) ? layer->id : -1));
	}
	bool Terminal::operator >(const Terminal & r) const {
		return (std::make_tuple(x, y, (layer) ? layer->id : -1) < std::make_tuple(r.x, r.y, (layer) ? layer->id : -1));
	}
	Neuron::Neuron(const NeuronFunction& func,bool _bias,float val) :transform(func),value(val) {
		if (_bias) {
			bias.reset(new Bias());
			SignalPtr bsignal=SignalPtr(new Signal());
			bsignal->value = 0.5f;
			addInput(bsignal);
			bias->addOutput(bsignal);
		}
	}
	float Neuron::evaluate() {
		float sum1 = 0.0f, sum2=0.0f;
		if (transform.type() != NeuronFunctionType::Constant) {
			for (SignalPtr sig : input) {
				sum2 = 0.0f;
				for (Neuron* inner : sig->input) {
					sum2 += inner->value;
				}
				sum1 += sig->value*sum2;
			}
			value = transform.forward(sum1);
		}
		return value;
	}
}