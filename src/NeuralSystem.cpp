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
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
using namespace aly;
namespace tgr {
	void NeuralSystem::backpropagate() {
		for (NeuralLayer* layer : backpropLayers) {
			layer->backpropagate();
		}
	}
	bool NeuralSystem::optimize() {
		bool ret = false;
		for (NeuralLayer* layer : backpropLayers) {
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
				float err = neuron->value - output(i, j).x;
				neuron->change += err;

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
			float err = neuron->value - output[i];
			neuron->change += err;
			residual += err*err;
		}
		residual /= double(output.size());
		layer->accumulate(residual);
		return layer->getResidual();
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
	NeuralSystem::NeuralSystem(const std::shared_ptr<aly::NeuralFlowPane>& pane) :flowPane(pane) {

	}

	void NeuralSystem::evaluate() {
		for (NeuralLayerPtr layer : layers) {
			layer->evaluate();
			Vector1f data = layer->toVector();
			//std::cout << layer->getName() << ":: Sum: " << data.sum() << " Std. Dev: " << data.stdDev() << std::endl;
		}
	}
	void NeuralSystem::initialize() {
		roots.clear();
		leafs.clear();
		std::list<NeuralLayerPtr> q;
		std::list<NeuralLayer*> q2;
		for (NeuralLayerPtr layer : layers) {
			layer->update();
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
			for (NeuralLayerPtr child : layer->getChildren()) {
				if (child->visitedDependencies()) {
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
				if (dep->visitedChildren()) {
					q2.push_back(dep);
				}
			}
		}
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
	}
	void NeuralSystem::initialize(const aly::ExpandTreePtr& tree) {
		initialize();
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
	void WriteNeuralSystemToFile(const std::string& file, const NeuralSystem& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ofstream os(file);
			cereal::JSONOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else if (ext == "xml") {
			std::ofstream os(file);
			cereal::XMLOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else {
			std::ofstream os(file, std::ios::binary);
			cereal::PortableBinaryOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
	}
	void ReadNeuralSystemFromFile(const std::string& file, NeuralSystem& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ifstream os(file);
			cereal::JSONInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else if (ext == "xml") {
			std::ifstream os(file);
			cereal::XMLInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else {
			std::ifstream os(file, std::ios::binary);
			cereal::PortableBinaryInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
	}

}