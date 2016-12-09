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
#include "NeuralFlowPane.h"

using namespace aly;
namespace tgr {
	void NeuralSystem::backpropagate() {
		for (int n = (int)filters.size() - 1; n >= 0; n--) {
			filters[n]->backpropagate();
		}
	}
	void NeuralSystem::setKnowledge(const NeuralKnowledge& k) {
		knowledge = k;
		for (NeuralLayerPtr layer : layers) {
			layer->set(k.getWeights(*layer), k.getBiasWeights(*layer));
		}
	}
	bool NeuralSystem::optimize() {
		bool ret = false;
		for (NeuralLayerPtr layer : layers) {
			ret |= layer->optimize();
		}
		return ret;
	}
	void NeuralSystem::setOptimizer(const NeuralOptimizationPtr& opt) {
		for (auto layer : layers) {
			if (layer->isTrainable()) {
				layer->setOptimizer(opt);
			}
		}
	}
	double NeuralSystem::accumulate(const NeuralLayerPtr& layer, const Image1f& output) {
		double residual = 0;
		for (int j = 0; j < output.height; j++) {
			for (int i = 0; i < output.width; i++) {
				Neuron* neuron = layer->get(i, j);
				float err = *neuron->value - output(i, j).x;
				*neuron->change += err;

				residual += std::abs(err);
			}
		}
		residual /= double(output.size());
		layer->accumulate(residual);
		return layer->getResidual();
	}
	double NeuralSystem::accumulate(const NeuralLayerPtr& layer, const std::vector<float>& output) {
		double residual = 0;
		for (size_t i = 0; i < output.size(); i++) {
			Neuron* neuron = layer->get(i);
			float err = *neuron->value - output[i];
			*neuron->change = err*neuron->forwardChange(*neuron->value);
			//std::cout << i << ": " << *neuron->change << " " << *neuron->value <<" "<<err<<" "<< output[i]<< std::endl;
			residual += err*err;
		}
		residual /= double(output.size());
		layer->accumulate(residual);
		return residual;
	}
	void NeuralSystem::reset() {
		for (NeuralLayerPtr layer : layers) {
			layer->reset();
		}
	}
	void NeuralSystem::setLayer(const NeuralLayerPtr& layer, const Image1f& input) {
		layer->set(input);
	}
	void NeuralSystem::setLayer(const NeuralLayerPtr& layer, const std::vector<float>& input) {
		layer->set(input);
	}
	void NeuralSystem::getLayer(const NeuralLayerPtr& layer, Image1f& input) {
		layer->get(input);
	}
	void NeuralSystem::getLayer(const NeuralLayerPtr& layer, std::vector<float>& input) {
		layer->get(input);
	}
	NeuralSystem::NeuralSystem(const std::shared_ptr<aly::NeuralFlowPane>& pane) :flowPane(pane),initialized(false) {

	}
	void NeuralSystem::evaluate() {
		if (!initialized)initialize();
		for (NeuralFilterPtr filter:filters) {
			filter->evaluate();
		}
	}
	NeuralKnowledge& NeuralSystem::updateKnowledge() {
		knowledge.set(*this);
		return knowledge;
	}
	void NeuralSystem::initialize() {
		roots.clear();
		leafs.clear();
		std::list<NeuralLayerPtr> q;
		for (NeuralLayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isRoot()) {
				roots.push_back(layer);
				q.push_back(layer);
			}
		}
		std::vector<NeuralLayerPtr> order;
		int index = 0;
		while (!q.empty()) {
			NeuralLayerPtr layer = q.front();
			if (!layer->isVisited())layer->compile();
			q.pop_front();
			layer->setId(index++);
			layer->setVisited(true);
			order.push_back(layer);
			for (NeuralLayerPtr child : layer->getChildren()) {
				if (child->visitedDependencies()) {
					q.push_back(child);
				}
			}
		}
		layers = order;
		order.clear();
		for (NeuralLayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isLeaf()) {
				leafs.push_back(layer);
			}
		}
		initializeWeights(0.0f, 1.0f);
		knowledge.set(*this);
		initialized = true;
	}
	void NeuralSystem::initializeWeights(float minW, float maxW) {
		for (NeuralLayerPtr layer : layers) {
			layer->initializeWeights(minW, maxW);
		}
	}
	void NeuralSystem::add(const std::shared_ptr<NeuralFilter>& filter, const NeuronFunction& func) {
		filter->initialize(*this, func);
		auto inputs = filter->getInputLayers();
		auto output = filter->getOutputLayers();
		for (auto layer : inputs) {
			layer->setSystem(this);
		}
		for (auto layer : output) {
			layer->setSystem(this);
		}
		layers.insert(layers.end(), inputs.begin(), inputs.end());
		layers.insert(layers.end(), output.begin(), output.end());
		filters.push_back(filter);
		initialized = false;

	}
	void NeuralSystem::initialize(const aly::ExpandTreePtr& tree) {
		TreeItemPtr root = TreeItemPtr(new TreeItem("Neural Layers"));
		tree->addItem(root);
		root->setExpanded(true);
		for (NeuralLayerPtr n : roots) {
			n->initialize(tree, root);
		}
	}
	Neuron* NeuralSystem::getNeuron(const Terminal& t) const {
		return t.layer->get(t.x, t.y);
	}


}