#include "NeuralFilter.h"
namespace tgr {
	void NeuralFilter::evaluate() {
		for (NeuralLayerPtr layer : outputLayers) {
			layer->evaluate();
		}
	}
	void NeuralFilter::backpropagate() {
		for (NeuralLayerPtr layer : inputLayers) {
			layer->backpropagate();
		}
		for (NeuralLayerPtr layer : outputLayers) {
			if(layer->isLeaf())layer->backpropagate();
		}
	}
}