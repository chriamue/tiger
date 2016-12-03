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
#include <AlloyOptimizationMath.h>
#include "NeuronFunction.h"
#include <memory>
#include <map>
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
	class Signal {
	protected:
		float* weight;
		float* change;
	public:
		std::map<const Neuron*,std::vector<Neuron*>> mapping;
		Signal() :weight(nullptr),change(nullptr) {

		}
		void setWeightPointer(float* ptr) {
			weight = ptr;
		}
		void setChangePointer(float* ptr) {
			change = ptr;
		}
		float* getWeight() {
			return weight;
		}
		float getWeightValue() {
			return *weight;
		}
		float getChangeValue() {
			return *change;
		}
		void setWeight(float w) {
			if (!weight)throw std::runtime_error("weight pointer missing.");
			*weight=w;
		}
		float* getChange() {
			return change;
		}
		void setChange(float w) {
			if (!change)throw std::runtime_error("change pointer missing.");
			*change = w;
		}
		std::vector<Neuron*>& operator[](const Neuron* n) {
			return mapping.at(n);
		}
		std::vector<Neuron*>& get(const Neuron* n) {
			return mapping.at(n);
		}
		const std::vector<Neuron*>& operator[](const Neuron* n) const {
			return mapping.at(n);
		}
		const std::vector<Neuron*>& get(const Neuron* n) const {
			return mapping.at(n);
		}
		size_t size(const Neuron* n) const {
			return mapping.at(n).size();
		}
		
		void add(Neuron* in,const Neuron* out) {
			mapping[out].push_back(in);
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
		float change;
		bool active;
		friend class NeuralLayer;
		float normalizedValue() const {
			return aly::clamp((value - transform.min()) / std::max(1E-10f,transform.max() - transform.min()),0.0f,1.0f);
		}
		float forward(float val) const {
			return transform.forward(val);
		}
		float forwardChange(float val) const {
			return transform.change(val);
		}
		size_t getInputWeightSize() const {
			return input.size();
		}
		size_t getOutputWeightSize() const {
			return output.size();
		}
		size_t getInputNeuronSize() const {
			size_t count = 0;
			for (auto in : input) {
				count += in->size(this);
			}
			return count;
		}
		std::string getType() const;
		size_t getOutputNeuronSize() const {
			return output.size();
		}
		float evaluate();
		float backpropagate();
		const std::vector<SignalPtr>& getInput() const {
			return input;
		}
		const std::vector<SignalPtr>& getOutput() const {
			return output;
		}
		std::vector<SignalPtr>& getInput(){
			return input;
		}
		std::vector<SignalPtr>& getOutput() {
			return output;
		}
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
		std::vector<const Neuron*> getOutputNeurons() const;
		void getOutputNeurons(std::vector<const Neuron*>& out) const;
		
		void addInput(const SignalPtr& s) {
			input.push_back(s);
		}
		void addOutput(const SignalPtr& s) {
			output.push_back(s);
		}
		float getInputWeight(size_t idx) {
			return *input[idx]->getWeight();
		}

		float getOutputWeight(size_t idx) {
			return *output[idx]->getWeight();
		}

		Neuron(const NeuronFunction& func = ReLU(),float val=0.0f);
		void setFunction(const NeuronFunction& func) {
			transform = func;
		}
	};
	class Bias : public Neuron {
		public:		
			Bias(float val=1.0f) :Neuron(Constant(&value), val) {

			}
	};
	void MakeConnection(Neuron* src, const std::shared_ptr<Signal>& signal,Neuron* dest);
	std::shared_ptr<Signal> MakeConnection(Neuron* src, Neuron* dest);
}
#endif