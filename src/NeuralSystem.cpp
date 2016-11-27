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
#include "NeuralSystem.h"
#include "NeuralFilter.h"
using namespace aly;
namespace tgr {
	void NeuralSystem::backpropagate() {
		for (NeuralLayer* layer : backpropLayers) {
			layer->evaluate();
		}
	}
	void NeuralSystem::pushInput(){
		for (const std::pair<Terminal, float>& pr : input) {
			Terminal t = pr.first;
			NeuralLayer* layer = t.layer;
			(*layer)(t.x, t.y).value = pr.second;
		}
	}

	void NeuralSystem::pullOutput(){
		for (auto iter = output.begin(); iter != output.end(); iter++) {
			Terminal t = iter->first;
			NeuralLayer* layer = t.layer;
			iter->second = (*layer)(t.x, t.y).value;
		}
	}
	void NeuralSystem::resetError() {
		for (auto iter = output.begin(); iter != output.end(); iter++) {
			Terminal t = iter->first;
			NeuralLayer* layer = t.layer;
			Neuron& neuron = (*layer)(t.x, t.y);
			neuron.change =0.0f;
		}
	}
	void NeuralSystem::accumulateError() {
		for (auto iter = output.begin(); iter != output.end(); iter++) {
			Terminal t = iter->first;
			NeuralLayer* layer = t.layer;
			Neuron& neuron = (*layer)(t.x, t.y);
			neuron.change += iter->second-neuron.value;
		}
	}
	void NeuralSystem::evaluate() {
		for (NeuralLayerPtr layer:layers) {
			layer->evaluate();
		}
	}
	void NeuralSystem::initialize() {
		roots.clear();
		leafs.clear();
		std::list<NeuralLayerPtr> q;
		std::list<NeuralLayer*> q2;
		for (NeuralLayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isRoot()) {
				roots.push_back(layer);
				q.push_back(layer);
			}
		}
		
		backpropLayers.clear();
		std::vector<NeuralLayerPtr> order;
		int index = 0;
		while (!q.empty()) {
			NeuralLayerPtr layer = q.front();
			q.pop_front();
			layer->id = index++;
			layer->setVisited(true);
			order.push_back(layer);
			for (NeuralLayerPtr child:layer->getChildren()) {
				if (child->ready()) {
					q.push_back(child);
				}
			}
		}
		layers = order;
		order.clear();

		q2.clear();
		for (NeuralLayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isLeaf()) {
				leafs.push_back(layer);
				q2.push_back(layer.get());
			}
		}

		while (!q2.empty()) {
			NeuralLayer* layer = q2.front();
			q2.pop_front();
			layer->setVisited(true);
			backpropLayers.push_back(layer);
			for (NeuralLayer* dep : layer->getDependencies()) {
				if (dep->ready()) {
					q2.push_back(dep);
				}
			}
		}
	}
	void NeuralSystem::add(const std::shared_ptr<NeuralFilter>& filter) {
		filter->initialize(*this);
		auto inputs= filter->getInputLayers();
		auto output = filter->getOutputLayers();
		layers.insert(layers.end(), inputs.begin(), inputs.end());
		layers.insert(layers.end(), output.begin(), output.end());
	}
	void NeuralSystem::initialize(const aly::ExpandTreePtr& tree) {
		initialize();
		TreeItemPtr root=TreeItemPtr(new TreeItem("Neural Layers"));
		tree->addItem(root);
		root->setExpanded(true);
		for (NeuralLayerPtr n : roots) {
			n->initialize(tree, root);
		}
	}
	void NeuralSystem::setInput(const Terminal& t, float value) {
		input[t] = value;
	}
	float NeuralSystem::getOutput(const Terminal& t) {
		return output[t];
	}
	Terminal NeuralSystem::addInput(int i, int j, const NeuralLayerPtr& layer, float value) {
		Terminal t(i, j,  layer.get());
		input[t] = value;
		Neuron& neuron = (*layer)(t.x, t.y);
		neuron.setFunction(Constant(&neuron.value));
		return t;
	}
	Terminal NeuralSystem::addOutput(int i, int j, const NeuralLayerPtr& layer, float value){
		Terminal t(i, j, layer.get());
		output[t] = value;
		return t;
	}


	Neuron* NeuralSystem::getNeuron(const Terminal& t) const {
		return t.layer->get(t.x, t.y);
	}
	SignalPtr NeuralSystem::add(Terminal source, Terminal target,float weight) {
		SignalPtr signal = std::shared_ptr<Signal>(new Signal(weight));
		Neuron* dest=getNeuron(target);
		Neuron* src = getNeuron(source);
		MakeConnection(src, signal,dest);

		return signal;
	}
	void NeuralSystem::train(float learningRate) {

	}
}