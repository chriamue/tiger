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
#include "AveragePoolFilter.h"
#include "AlloyMath.h"
using namespace aly;
namespace tgr {
	AveragePoolFilter::AveragePoolFilter(TigerApp* app,const std::vector<NeuralLayerPtr>& inputLayers, int kernelSize):NeuralFilter(app,"Average Pool"),kernelSize(kernelSize) {
		NeuralFilter::inputLayers = inputLayers;
		for (NeuralLayerPtr layer : inputLayers) {
			if (layer->width%kernelSize != 0 || layer->height%kernelSize != 0) {
				throw std::runtime_error("Map size must be divisible by kernel size.");
			}
		}
	}
	AveragePoolFilter::AveragePoolFilter(TigerApp* app, const NeuralLayerPtr& inputLayer, int kernelSize) :NeuralFilter(app, "Average Pool"), kernelSize(kernelSize) {
		NeuralFilter::inputLayers.push_back(inputLayer);
		for (NeuralLayerPtr layer : inputLayers) {
			if (layer->width%kernelSize != 0 || layer->height%kernelSize != 0) {
				throw std::runtime_error("Map size must be divisible by kernel size.");
			}
		}
	}
	void AveragePoolFilter::initialize(NeuralSystem& sys) {
		std::vector<SignalPtr> signals;
		outputLayers.resize(inputLayers.size());
		for (int k = 0; k < inputLayers.size(); k++) {
			NeuralLayerPtr inputLayer = inputLayers[k];
			outputLayers[k] = NeuralLayerPtr(new NeuralLayer(app,name,inputLayer->width/kernelSize,inputLayer->height/kernelSize,1,true));
			NeuralLayerPtr outputLayer = outputLayers[k];
			inputLayer->addChild(outputLayer);
			for (int j = 0; j < outputLayer->height; j++) {
				for (int i = 0; i < outputLayer->width; i++) {
					SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1/(float)(kernelSize*kernelSize))));
					Neuron* dest = outputLayer->get(i, j);
					dest->addInput(sig);
					for (int jj = 0; jj < kernelSize; jj++) {
						for (int ii = 0; ii < kernelSize; ii++) {
							Neuron* src=inputLayer->get(i*kernelSize + ii, j*kernelSize + jj);
							src->addOutput(sig);
							sig->add(src, dest);
						}
					}
				}
			}
		}

	}
}