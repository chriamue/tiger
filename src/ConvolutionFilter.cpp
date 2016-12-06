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
	ConvolutionFilter::ConvolutionFilter(int width, int height, int kernelSize, int features, bool bias) :NeuralFilter("Feature"), kernelSize(kernelSize), bias(bias) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(NeuralLayerPtr(new NeuralLayer("Input Layer", width, height, 1, false, Linear())));
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(const NeuralLayerPtr& layer, int kernelSize, int features, bool bias) :NeuralFilter("Feature"), kernelSize(kernelSize), bias(bias) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers.push_back(layer);
		outputLayers.resize(features);
	}
	ConvolutionFilter::ConvolutionFilter(const std::vector<NeuralLayerPtr>& layers, int kernelSize, int features, bool bias) :NeuralFilter("Feature"), kernelSize(kernelSize), bias(bias) {
		if (kernelSize % 2 == 0) {
			throw std::runtime_error("Kernel size must be odd.");
		}
		inputLayers = layers;
		outputLayers.resize(features);
	}
	void ConvolutionFilter::evaluate() {
		/*
		int pad = kernelSize / 2;
		int width = inputLayers[0]->width;
		int height = inputLayers[0]->height;
		int ow = width - 2 * pad;
		int oh = height - 2 * pad;
		int K = kernelSize*kernelSize;
		const Knowledge& inResponses = inputLayers.front()->responses;
		for (NeuralLayerPtr layer : outputLayers) {
			const Knowledge& weights = layer->weights;
			Knowledge& outResponses = layer->responses;
			std::cout << "Layer: " << layer->getName() << std::endl;
			std::cout << "Input Weights " << weights.size() << std::endl;
			std::cout << "Input Responses " << inResponses.size() << std::endl;
			std::cout << "Output Responses " << outResponses.size() << std::endl;
//#pragma omp parallel for

			for (int j = 0; j < oh; j++) {
				for (int i = 0; i < ow; i++) {
					float sum = 0;
					float wsum = 0;
					float rsum = 0;
					int idx = (i + j*ow);
					for (int jj = 0; jj < kernelSize; jj++) {
						for (int ii = 0; ii < kernelSize; ii++) {
							float w = weights[idx + ii + kernelSize*jj];
							float r = inResponses[(i + ii) + width*(j + jj)];
							sum += r*w;
							wsum += w;
						}
					}
					//std::cout << "OUT " << sum << " " << wsum <<" "<<rsum<< std::endl;
					outResponses[i + j*ow] = transform.forward(sum);
				}
			}
			
		}
		*/
		NeuralFilter::evaluate();
	}
	void ConvolutionFilter::backpropagate() {
		NeuralFilter::backpropagate();
		/*
		for (NeuralLayerPtr layer : inputLayers) {
			layer->backpropagate();
		}
		for (NeuralLayerPtr layer : outputLayers) {
			if (layer->isLeaf())layer->backpropagate();
		}
		*/
	}
	void ConvolutionFilter::initialize(NeuralSystem& system, const NeuronFunction& func) {
		transform = func;
		int pad = kernelSize / 2;

		int width = inputLayers[0]->width;
		int height = inputLayers[0]->height;
		int ow = width - 2 * pad;
		int oh = height - 2 * pad;
		std::vector<SignalPtr> signals(kernelSize*kernelSize);
		int index = 0;
		if (connectionMap.size() == 0) {
			for (int f = 0; f < outputLayers.size(); f++) {
				NeuralLayerPtr outputLayer = NeuralLayerPtr(new NeuralLayer(MakeString() << name << " [" << f << "]", ow, oh, 1, bias, func));
				outputLayers[f] = outputLayer;
				index = 0;
				for (int jj = 0; jj < kernelSize; jj++) {
					for (int ii = 0; ii < kernelSize; ii++) {
						SignalPtr sig = SignalPtr(new Signal());
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
					outputLayer = NeuralLayerPtr(new NeuralLayer(MakeString() << name << " [" << outIdx << "]", ow, oh, 1, bias, func));
					outputLayers[outIdx] = outputLayer;
				}
				else {
					outputLayer = outputLayers[outIdx];
				}
				index = 0;
				for (int jj = 0; jj < kernelSize; jj++) {
					for (int ii = 0; ii < kernelSize; ii++) {
						SignalPtr sig = SignalPtr(new Signal());
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