#include "ConvolutionFilter.h"
#include "AlloyMath.h"

using namespace aly;
namespace tgr {
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, int width, int height, int kernelSize, int features) :NeuralFilter(app,"Feature"), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(NeuralLayerPtr(new NeuralLayer(app,"Input Layer", width, height)));
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, const NeuralLayerPtr& layer, int kernelSize, int features) :NeuralFilter(app, "Feature"), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(layer);
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, const std::vector<NeuralLayerPtr>& layers, int kernelSize, int features) :NeuralFilter(app, "Feature"), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers = layers;
		outputLayers.resize(features);
	}
	void ConvolutionFilter::initialize(NeuralSystem& system) {
		int pad = kernelSize / 2;
		
		int width = inputLayers[0]->width;
		int height = inputLayers[0]->height;
		int ow = width - 2 * pad;
		int oh = height - 2 * pad;
		std::vector<SignalPtr> signals(kernelSize*kernelSize);
		int index = 0;
		for (int jj = 0; jj < kernelSize; jj++) {
			for (int ii = 0; ii < kernelSize; ii++) {
				SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1.0f)));
				signals[index++]=sig;
			}
		}
		for (int f = 0; f < outputLayers.size(); f++) {
			NeuralLayerPtr outputLayer = NeuralLayerPtr(new NeuralLayer(app, MakeString() << name<< " [" << f << "]", ow, oh));
			outputLayers[f] = outputLayer;
			outputLayer->setFunction(Tanh());
			for (NeuralLayerPtr inputLayer : inputLayers) {
				inputLayer->addChild(outputLayer);
				for (int j = 0; j < oh; j++) {
					for (int i = 0; i < ow; i++) {
						index = 0;
						for (int jj = 0; jj < kernelSize; jj++) {
							for (int ii = 0; ii < kernelSize; ii++) {
								MakeConnection(inputLayer->get(i+ii, j+jj),signals[index++], outputLayer->get(i, j));
							}
						}
					}
				}
			}
		}
	}
}