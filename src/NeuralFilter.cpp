#include "NeuralFilter.h"
#include "AlloyMath.h"
using namespace aly;
namespace tgr {
	ConvolutionFilter::ConvolutionFilter(int width, int height, int kernelX, int kernelY, int features) :NeuralFilter(), kernelX(kernelX), kernelY(kernelY) {
		if (kernelX % 2 == 0 || kernelY % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayer = NeuralLayerPtr(new NeuralLayer(MakeString()<<"convolution ["<<kernelX<<"x"<<kernelY<<"]",width, height));
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(const NeuralLayerPtr& layer, int kernelX, int kernelY, int features) :NeuralFilter(),kernelX(kernelX), kernelY(kernelY) {
		if (kernelX % 2 == 0 || kernelY % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayer = layer;
		outputLayers.resize(features);
	}
	void ConvolutionFilter::attach(NeuralSystem& system) {
		int padX = kernelX / 2;
		int padY = kernelY / 2;
		int width = inputLayer->width;
		int height = inputLayer->height;
		int ow = width - 2 * padX;
		int oh = height - 2 * padY;
		std::vector<SignalPtr> signals;
		for (int jj = 0; jj < kernelY; jj++) {
			for (int ii = 0; ii < kernelX; ii++) {
				SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1.0f)));
				signals.push_back(sig);
				system.add(sig);
			}
		}
		for (int f = 0; f < outputLayers.size(); f++) {
			outputLayers[f] = NeuralLayerPtr(new NeuralLayer(MakeString() << "feature [" << f << "]", ow, oh));
			NeuralLayerPtr outputLayer = outputLayers[f];
			outputLayer->setFunction(Tanh());
			inputLayer->addChild(outputLayer);
			for (int j = 0; j < oh; j++) {
				for (int i = 0; i < ow; i++) {
					int index = 0;
					for (int jj = 0; jj < kernelY; jj++) {
						for (int ii = 0; ii < kernelX; ii++) {
							Neuron& src = inputLayer->get(i + ii, j + jj);
							Neuron& tar = outputLayer->get(i, j);
							SignalPtr sig = signals[index++];
							src.addOutput(sig);
							tar.addInput(sig);
						}
					}
				}
			}

		}
	}
}