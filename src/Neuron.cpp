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
		return (std::make_tuple(x, y, (layer) ? layer->getId() : -1) < std::make_tuple(r.x, r.y, (layer) ? layer->getId() : -1));
	}
	bool Terminal::operator >(const Terminal & r) const {
		return (std::make_tuple(x, y, (layer) ? layer->getId() : -1) < std::make_tuple(r.x, r.y, (layer) ? layer->getId() : -1));
	}
	Neuron::Neuron(const NeuronFunction& func) :transform(func),value(nullptr),change(nullptr),active(false) {
	}
	std::vector<Neuron*> Neuron::getInputNeurons()  const {
		std::vector<Neuron*> out;
		for (SignalPtr sig : input) {
			auto in = sig->get(this);
				out.insert(out.end(), in.begin(), in.end());
		}
		return out;
	}
	void Neuron::getInputNeurons(std::vector<Neuron*>& out) const {
		out.clear();
		for (SignalPtr sig : input) {
			auto in = sig->get(this);
			out.insert(out.end(), in.begin(), in.end());
		}
	}
	std::vector<const Neuron*> Neuron::getOutputNeurons() const {
		std::vector<const Neuron*> out;
		for (SignalPtr sig : output) {
			for (auto pr : sig->mapping) {
				out.push_back(pr.first);
			}
		}
		return out;
	}
	void Neuron::getOutputNeurons(std::vector<const Neuron*>& out) const {
		out.clear();
		for (SignalPtr sig : output) {
			for (auto pr : sig->mapping) {
				out.push_back(pr.first);
			}
		}
	}
	std::string Neuron::getType() const {
		return aly::MakeString() << transform.type();
	}
	float Neuron::backpropagate() {
		float sum1 = 0.0f,sum2;
		int count = 0;
		if (output.size() > 0) {
			//std::cout << "dw= [";
			for (SignalPtr sig : output) {
				sum2 = 0.0f;
				for (auto pr : sig->mapping) {
					sum2 += *pr.first->change;
					count++;
				}
				sum1 += *sig->weight*sum2;
				//std::cout << std::setfill(' ') << std::setw(5) << aly::round(sig->value, 3) << " * [" << aly::round(sum2, 3) << "] ";
			}
			*change=sum1*transform.change(*value) / count;
			//std::cout << "] " << change << std::endl;
		}
		for (SignalPtr sig : input) {
			sum2 = 0.0f;
			std::vector<Neuron*>& input2 = sig->get(this);
			for (Neuron* inner : input2) {
				sum2 += *inner->value;
			}
			*sig->change += *change*sum2 / (input2.size());
		}
		return *change;
	}

	float Neuron::evaluate() {
		float sum1 = 0.0f, sum2;
		int count = 0;
		
		if (input.size() > 0) {
			//std::cout << "w= [";
			for (SignalPtr sig : input) {
				sum2 = 0.0f;
				std::vector<Neuron*>& input2 = sig->get(this);
				for (Neuron* inner : input2) {
					sum2 += *inner->value;
					count++;
				}
				sum1 += (*sig->weight)*sum2;
				//std::cout <<std::setfill(' ')<< std::setw(5) << aly::round(sig->value,3) << " * [" << aly::round(sum2, 3) << "] ";
			}
			*change = 0.0f;
			sum1 /= count;
			*value = transform.forward(sum1);
			//std::cout << "] T(" <<sum1<<") = "<< value << std::endl;
		}
		
		return *value;
	}
	void MakeConnection(Neuron* src,const std::shared_ptr<Signal>& signal, Neuron* dest) {
		src->addOutput(signal);
		dest->addInput(signal);
		signal->add(src,dest);
	}
	std::shared_ptr<Signal> MakeConnection(Neuron* src, Neuron* dest) {
		std::shared_ptr<Signal> signal(new Signal());
		src->addOutput(signal);
		dest->addInput(signal);
		signal->add(src, dest);
		return signal;
	}
}