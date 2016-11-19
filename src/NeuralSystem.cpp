#include "NeuralSystem.h"
namespace tgr {
	void NeuralSystem::add(const NeuronLayerPtr& layer) {
		layer->setId((int)layers.size());
		layers.push_back(layer);
	}
	void NeuralSystem::evaluate() {
		int N = (int)std::min(inputTerminals.size(), inputs.size());
#pragma omp parallel for;
		for (int i = 0; i < N; i++) {
			Terminal t=inputTerminals[i];
			NeuronLayerPtr layer = layers[t.layer];
			(*layer)(t.x, t.y, t.bin).value = inputs[i];
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
	void NeuralSystem::add(const SignalPtr& signal) {
		signals.push_back(signal);
	}
	SignalPtr NeuralSystem::add(Terminal source, Terminal target,float weight) {
		SignalPtr sig = std::shared_ptr<Signal>(new Signal(source,target,weight));
		signals.push_back(sig);
		return sig;
	}
	void NeuralSystem::train(float learningRate) {

	}
}