#include "ConvolutionFilter.h"
#include "AlloyMath.h"

using namespace aly;
namespace tgr {
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, int width, int height, int kernelSize, int features) :NeuralFilter(app), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(NeuralLayerPtr(new NeuralLayer(app, MakeString() << "convolution [" << kernelSize << "]", width, height)));
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, const NeuralLayerPtr& layer, int kernelSize, int features) :NeuralFilter(app), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(layer);
		outputLayers.resize(features);
	}
	void ConvolutionFilter::initialize(NeuralSystem& system) {
		int pad = kernelSize / 2;
		NeuralLayerPtr inputLayer = inputLayers[0];
		int width = inputLayer->width;
		int height = inputLayer->height;
		int ow = width - 2 * pad;
		int oh = height - 2 * pad;
		std::vector<SignalPtr> signals;
		for (int jj = 0; jj < kernelSize; jj++) {
			for (int ii = 0; ii < kernelSize; ii++) {
				SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1.0f)));
				signals.push_back(sig);
				system.add(sig);
			}
		}
		for (int f = 0; f < outputLayers.size(); f++) {
			outputLayers[f] = NeuralLayerPtr(new NeuralLayer(app, MakeString() << "feature [" << f << "]", ow, oh));
			NeuralLayerPtr outputLayer = outputLayers[f];
			outputLayer->setFunction(Tanh());
			inputLayer->addChild(outputLayer);
			for (int j = 0; j < oh; j++) {
				for (int i = 0; i < ow; i++) {
					int index = 0;
					for (int jj = 0; jj < kernelSize; jj++) {
						for (int ii = 0; ii < kernelSize; ii++) {
							Neuron& src = inputLayer->get(i + ii, j + jj);
							Neuron& dest = outputLayer->get(i, j);
							SignalPtr sig = signals[index++];
							src.addOutput(sig);
							dest.addInput(sig);
						}
					}
				}
			}

		}
	}
}