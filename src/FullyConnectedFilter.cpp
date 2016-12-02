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
#include "FullyConnectedFilter.h"
#include "AlloyMath.h"
using namespace aly;
namespace tgr {
	FullyConnectedFilter::FullyConnectedFilter(const std::string& name, const std::vector<NeuralLayerPtr>& inputLayers, int width, int height, bool bias) :NeuralFilter(name), width(width), height(height), bias(bias) {
		NeuralFilter::inputLayers = inputLayers;
	}
	FullyConnectedFilter::FullyConnectedFilter(const std::string& name, const NeuralLayerPtr& inputLayer, int width, int height, bool bias) : NeuralFilter(name), width(width), height(height), bias(bias) {
		NeuralFilter::inputLayers.push_back(inputLayer);
	}
	FullyConnectedFilter::FullyConnectedFilter(const std::string& name, int inWidth,int inHeight,int width, int height, bool bias) : NeuralFilter(name), width(width),height(height),bias(bias) {
		inputLayers.push_back(NeuralLayerPtr(new NeuralLayer("Input Layer",inWidth,inHeight, 1,false, Tanh())));
	}
	void FullyConnectedFilter::initialize(NeuralSystem& sys, const NeuronFunction& func) {
		std::vector<SignalPtr> signals;
		outputLayers.push_back(NeuralLayerPtr(new NeuralLayer( name, width, height, 1, bias, func)));
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