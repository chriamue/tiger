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
	int64_t Signal::ID_COUNT = 0;
	std::vector<Neuron*> Neuron::getInputNeurons()  const {
		std::vector<Neuron*> out;
		for (SignalPtr sig : input) {
			auto in = sig->getForward(this);
				out.insert(out.end(), in.begin(), in.end());
		}
		return out;
	}
	void Neuron::getInputNeurons(std::vector<Neuron*>& out) const {
		out.clear();
		for (SignalPtr sig : input) {
			auto in = sig->getForward(this);
			out.insert(out.end(), in.begin(), in.end());
		}
	}
	std::vector<const Neuron*> Neuron::getOutputNeurons() const {
		std::vector<const Neuron*> out;
		for (SignalPtr sig : output) {
			for (auto pr : sig->forwardMapping) {
				out.push_back(pr.first);
			}
		}
		return out;
	}
	void Neuron::getOutputNeurons(std::vector<const Neuron*>& out) const {
		out.clear();
		for (SignalPtr sig : output) {
			for (auto pr : sig->forwardMapping) {
				out.push_back(pr.first);
			}
		}
	}
	void Neuron::print() {
		/*
		std::cout << "====== Neuron " << id << " ======" << std::endl;
		std::cout << "Input: " << input.size() << " Output: " << output.size() << std::endl;
		for (SignalPtr sig : output) {
			std::cout << "***** Output Signal: " << sig->id << std::endl;
			for (Neuron* inner : sig->getBackward(this)) {
				std::cout << "***** Output Neuron: " << inner->id << std::endl;
			}
		}
		for (SignalPtr sig : input) {
			std::cout << "+++++ Input Signal: " << sig->id << std::endl;
			for (Neuron* inner : sig->getForward(this)) {
				std::cout << "+++++ Input Neuron: " << inner->id << std::endl;
			}
		}
		*/
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
				for (Neuron* outer : sig->getBackward(this)) {
					sum2 += *outer->change;
					count++;
				}
				sum1 += *sig->weight*sum2;
				//std::cout << std::setfill(' ') << std::setw(5) << aly::round(*sig->weight, 3) << " * [" << aly::round(sum2, 3) << "] ";
			}
			//Normalize change so that derivative doesn't blow up
			*change = sum1*transform.change(*value) / count;
		}
		//std::cout << "] " << change << std::endl;
		for (SignalPtr sig : input) {
			sum2 = 0.0f;
			for (Neuron* inner : sig->getForward(this)) {
				sum2 += *inner->value;
			}
			*sig->change += *change * sum2;
		}
		return *change;
	}

	float Neuron::evaluate() {
		float sum1 = 0.0f, sum2;
		int count = 0;
		if (input.size() > 0) {
			//if (debug)std::cout << "w= [";
			for (SignalPtr sig : input) {
				sum2 = 0.0f;
				for (Neuron* inner : sig->getForward(this)) {
					sum2 += *inner->value;
					count++;
				}
				sum1 += (*sig->weight)*sum2;
				//if(debug)std::cout <<std::setfill(' ')<< std::setw(5) << aly::round(*sig->weight,3) << " * [" << aly::round(sum2, 3) << "] ";
			}
			*change = 0.0f;
			sum1 /= count;
			*value = transform.forward(sum1);
			//if (debug)std::cout << "] T(" <<sum1<<") = "<< *value <<" "<< std::exp(2 * sum1)<< std::endl;
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