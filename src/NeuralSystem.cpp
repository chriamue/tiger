#include "NeuralSystem.h"
#include "NeuralFilter.h"
using namespace aly;
namespace tgr {

	void NeuralSystem::evaluate() {
		for (const std::pair<Terminal, float>& pr : input) {
			Terminal t = pr.first;
			NeuralLayer* layer= t.layer;
			(*layer)(t.x, t.y).value = pr.second;
		}
		for (NeuralLayerPtr layer:layers) {
			
			layer->evaluate();
		}
		for (auto iter = output.begin(); iter != output.end();iter++) {
			Terminal t = iter->first;
			NeuralLayer* layer = t.layer;
			iter->second=(*layer)(t.x, t.y).value;
		}
	}
	void NeuralSystem::initialize() {
		roots.clear();
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

	void NeuralSystem::add(const SignalPtr& signal) {
		signals.push_back(signal);
	}
	void NeuralSystem::add(const std::vector<SignalPtr>& sigs) {
		signals.insert(signals.end(), sigs.begin(), sigs.end());
	}
	Neuron* NeuralSystem::getNeuron(const Terminal& t) const {
		return t.layer->get(t.x, t.y);
	}
	SignalPtr NeuralSystem::add(Terminal source, Terminal target,float weight) {
		SignalPtr signal = std::shared_ptr<Signal>(new Signal(weight));
		getNeuron(target)->addInput(signal);
		getNeuron(source)->addOutput(signal);
		signals.push_back(signal);
		return signal;
	}
	void NeuralSystem::train(float learningRate) {

	}
}