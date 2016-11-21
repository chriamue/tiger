#include "NeuralSystem.h"
#include "NeuralFilter.h"
namespace tgr {

	void NeuralSystem::evaluate() {
		int N = (int)std::min(inputTerminals.size(), input.size());
		/*
#pragma omp parallel for;
		for (int i = 0; i < N; i++) {
			Terminal t=inputTerminals[i];
			NeuralLayerPtr layer = layers[t.layer];
			(*layer)(t.x, t.y).value = input[i];
		}
		*/
	}
	void NeuralSystem::initialize() {
		roots.clear();
		for (NeuralLayerPtr layer : layers) {
			if (layer->isRoot()) {
				roots.push_back(layer);
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
	void NeuralSystem::add(const SignalPtr& signal) {
		signals.insert(signal);
	}
	Neuron* NeuralSystem::getNeuron(const Terminal& t) const {
		return &t.layer->get(t.x, t.y);
	}
	SignalPtr NeuralSystem::connect(int si, int sj, const NeuralLayerPtr& sl, int ti, int tj, const NeuralLayerPtr& tl, float weight) {
		return add(Terminal(si, sj, 0, sl), Terminal(ti, tj, 0, tl),weight);
	}
	SignalPtr NeuralSystem::connect(int si, int sj,int sb, const NeuralLayerPtr& sl, int ti, int tj, int tb, const NeuralLayerPtr& tl, float weight) {
		return add(Terminal(si, sj, tb, sl), Terminal(ti, tj, tb, tl), weight);
	}
	SignalPtr NeuralSystem::add(Terminal source, Terminal target,float weight) {
		SignalPtr signal = std::shared_ptr<Signal>(new Signal(weight));
		getNeuron(target)->addInput(signal);
		getNeuron(source)->addOutput(signal);
		signals.insert(signal);
		return signal;
	}
	void NeuralSystem::train(float learningRate) {

	}
}