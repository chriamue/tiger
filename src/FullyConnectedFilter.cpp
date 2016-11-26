#include "FullyConnectedFilter.h"
#include "AlloyMath.h"
using namespace aly;
namespace tgr {
	FullyConnectedFilter::FullyConnectedFilter(TigerApp* app, const std::vector<NeuralLayerPtr>& inputLayers, int width, int height) :NeuralFilter(app, "Average Pool"), width(width), height(height) {
		NeuralFilter::inputLayers = inputLayers;
	}
	FullyConnectedFilter::FullyConnectedFilter(TigerApp* app, const NeuralLayerPtr& inputLayer, int width, int height) : NeuralFilter(app, "Fully Connected"), width(width), height(height) {
		NeuralFilter::inputLayers.push_back(inputLayer);
	}
	void FullyConnectedFilter::initialize(NeuralSystem& sys) {
		std::vector<SignalPtr> signals;
		outputLayers.push_back(NeuralLayerPtr(new NeuralLayer(app, name, width, height, 1, true)));
		NeuralLayerPtr outputLayer = outputLayers[0];
		for (int k = 0; k < inputLayers.size(); k++) {
			NeuralLayerPtr inputLayer = inputLayers[k];
			inputLayer->addChild(outputLayer);
			for (int j = 0; j < inputLayer->height; j++) {
				for (int i = 0; i < inputLayer->width; i++) {
					Neuron* src = inputLayer->get(i, j);
					for (int jj = 0; jj < outputLayer->height; jj++) {
						for (int ii = 0; ii < outputLayer->width; ii++) {
							SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1.0f)));
							Neuron* dest = outputLayer->get(ii, jj);
							MakeConnection(src, sig, dest);
						}
					}
				}
			}
		}
	}
}