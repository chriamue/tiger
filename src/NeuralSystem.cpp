#include "NeuralSystem.h"
namespace tgr {
	void NeuralSystem::add(const NeuronLayerPtr& layer) {
		layer->setId((int)layers.size());
		layers[layer->getId()]=(layer);
	}
	void NeuralSystem::evaluate() {
		int N = (int)std::min(inputTerminals.size(), input.size());
#pragma omp parallel for;
		for (int i = 0; i < N; i++) {
			Terminal t=inputTerminals[i];
			NeuronLayerPtr layer = layers[t.layer];
			(*layer)(t.x, t.y).value = input[i];
		}
	}
	void NeuralSystem::initialize() {
		int N = (int)inputTerminals.size();
		std::map<Terminal, std::vector<SignalPtr>> sourceMap;
		std::map<Terminal, std::vector<SignalPtr>> targetMap;

		for (SignalPtr signal : signals) {
			sourceMap[signal->source].push_back(signal);
		}
		for (SignalPtr signal : signals) {
			targetMap[signal->target].push_back(signal);
		}
		forwardNetwork.clear();
		forwardNetwork.reserve(layers.size() + 1);
		forwardNetwork.resize(1);
		for (int i = 0; i < N; i++) {
			Terminal t = inputTerminals[i];
			auto pos = sourceMap.find(t);
			if (pos != sourceMap.end()) {
				std::vector<SignalPtr>& pass = forwardNetwork[0];
				pass.insert(pass.end(), pos->second.begin(), pos->second.end());
			}
		}
		int previousLayer = 0;
		bool hasMore = false;
		do {
			std::vector<SignalPtr>& lastPass = forwardNetwork[previousLayer];
			std::set<Terminal> terms;
			for (SignalPtr signal : lastPass) {
				terms.insert(signal->target);
			}
			for (Terminal t:terms) {
				auto pos = sourceMap.find(t);
				if (pos != sourceMap.end()) {
					if (forwardNetwork.size() <= previousLayer + 1)forwardNetwork.push_back(std::vector<SignalPtr>());
					std::vector<SignalPtr>& pass = forwardNetwork[previousLayer+1];
					pass.insert(pass.end(), pos->second.begin(), pos->second.end());
					hasMore = true;
				}
			}
		} while(hasMore);
	}
	NeuronLayerPtr NeuralSystem::getLayer(int id) const {
		auto pos = layers.find(id);
		if (pos != layers.end()) {
			return pos->second;
		}
		else {
			return NeuronLayerPtr();
		}
	}
	void NeuralSystem::add(const SignalPtr& signal) {
		getNeuron(signal->target)->addInput(signal);
		getNeuron(signal->source)->addOutput(signal);
		signals.push_back(signal);
	}
	Neuron* NeuralSystem::getNeuron(const Terminal& t) const {
		return &getLayer(t.layer)->get(t.x, t.y);
	}
	SignalPtr NeuralSystem::connect(int si, int sj, const NeuronLayerPtr& sl, int ti, int tj, const NeuronLayerPtr& tl, float weight) {
		return add(Terminal(si, sj, 0, sl->getId()), Terminal(ti, tj, 0, tl->getId()),weight);
	}
	SignalPtr NeuralSystem::connect(int si, int sj,int sb, const NeuronLayerPtr& sl, int ti, int tj, int tb, const NeuronLayerPtr& tl, float weight) {
		return add(Terminal(si, sj, tb, sl->getId()), Terminal(ti, tj, tb, tl->getId()), weight);
	}
	SignalPtr NeuralSystem::add(Terminal source, Terminal target,float weight) {
		SignalPtr signal = std::shared_ptr<Signal>(new Signal(source,target,weight));
		getNeuron(target)->addInput(signal);
		getNeuron(source)->addOutput(signal);
		signals.push_back(signal);
		return signal;
	}
	void NeuralSystem::train(float learningRate) {

	}
}