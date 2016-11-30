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
#include "ConvolutionFilter.h"
#include "AlloyMath.h"

using namespace aly;
namespace tgr {
	ConvolutionFilter::ConvolutionFilter(TigerApp* app, int width, int height, int kernelSize, int features) :NeuralFilter(app, "Feature"), kernelSize(kernelSize) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(NeuralLayerPtr(new NeuralLayer(app, "Input Layer", width, height)));
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

		if (connectionMap.size() == 0) {
			for (int f = 0; f < outputLayers.size(); f++) {
				NeuralLayerPtr outputLayer = NeuralLayerPtr(new NeuralLayer(app, MakeString() << name << " [" << f << "]", ow, oh));
				outputLayers[f] = outputLayer;
				outputLayer->setFunction(Tanh());

				index = 0;
				for (int jj = 0; jj < kernelSize; jj++) {
					for (int ii = 0; ii < kernelSize; ii++) {
						SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f, 1.0f)));
						signals[index++] = sig;
					}
				}
				for (NeuralLayerPtr inputLayer : inputLayers) {
					inputLayer->addChild(outputLayer);
					for (int j = 0; j < oh; j++) {
						for (int i = 0; i < ow; i++) {
							index = 0;
							for (int jj = 0; jj < kernelSize; jj++) {
								for (int ii = 0; ii < kernelSize; ii++) {
									MakeConnection(inputLayer->get(i + ii, j + jj), signals[index++], outputLayer->get(i, j));
								}
							}
						}
					}
				}
			}
		}
		else {
			for (auto pr : connectionMap) {
				int inIdx = pr.first;
				int outIdx = pr.second;
				NeuralLayerPtr outputLayer;
				if (outputLayers[outIdx].get() == nullptr) {
					outputLayer = NeuralLayerPtr(new NeuralLayer(app, MakeString() << name << " [" << outIdx << "]", ow, oh));
					outputLayers[outIdx] = outputLayer;
					outputLayer->setFunction(Tanh());
				}
				else {
					outputLayer = outputLayers[outIdx];
				}
				index = 0;
				for (int jj = 0; jj < kernelSize; jj++) {
					for (int ii = 0; ii < kernelSize; ii++) {
						SignalPtr sig = SignalPtr(new Signal(RandomUniform(0.0f,1.0f)));
						signals[index++] = sig;
					}
				}
				NeuralLayerPtr inputLayer = inputLayers[inIdx];
				inputLayer->addChild(outputLayer);
				for (int j = 0; j < oh; j++) {
					for (int i = 0; i < ow; i++) {
						index = 0;
						for (int jj = 0; jj < kernelSize; jj++) {
							for (int ii = 0; ii < kernelSize; ii++) {
								MakeConnection(inputLayer->get(i + ii, j + jj), signals[index++], outputLayer->get(i, j));
							}
						}
					}
				}
			}
		}
	}
}