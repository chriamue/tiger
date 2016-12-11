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
		int pad = kernelSize / 2;
		int width = inputLayers[0]->width;
		int height = inputLayers[0]->height;
		int ow = width - 2 * pad;
		int oh = height - 2 * pad;
		int K = kernelSize*kernelSize;
		bool hasBias;
		const Knowledge& inResponses = inputLayers.front()->responses;
		for (NeuralLayerPtr layer : outputLayers) {
			const Knowledge& weights = layer->weights;
			const Knowledge& biasWeights = layer->biasWeights;
			Knowledge& outResponses = layer->responses;
			hasBias = (biasWeights.size() > 0);
#pragma omp parallel for
			for (int j = 0; j < oh; j++) {
				for (int i = 0; i < ow; i++) {
					int idx = i + j*ow;
					float sum = (hasBias) ? biasWeights[idx] : 0.0f;
					for (int jj = 0; jj < kernelSize; jj++) {
						for (int ii = 0; ii < kernelSize; ii++) {
							sum += weights[ii + kernelSize*jj] * inResponses[(i + ii) + width*(j + jj)];
						}
					}
					outResponses[idx] = transform.forward(sum / (weights.size() + biasWeights.size()));
				}
			}
			layer->setRegionDirty(true);
		}

	}
	void ConvolutionFilter::backpropagate() {
		NeuralFilter::backpropagate();
		/*
		std::vector<NeuralState> inStates(inputLayers.size());
		std::vector<NeuralState> outStates(outputLayers.size());
		NeuralLayerPtr inLayer = inputLayers.front();
		bool record = (inLayer->getName()=="Input Layer");
		if (record) {
			for (int n = 0; n < (int)inputLayers.size(); n++) {
				NeuralLayerPtr layer = inputLayers[n];
				NeuralState state = layer->getState();
				WriteNeuralStateToFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "orig_" << state.name << ".json", state);
				inStates[n] = state;
			}
			for (int n = 0; n < (int)outputLayers.size(); n++) {
				NeuralLayerPtr layer = outputLayers[n];
				NeuralState state = layer->getState();
				outStates[n] = state;
			}
			NeuralFilter::backpropagate();
			for (int n = 0; n < (int)inputLayers.size(); n++) {
				NeuralLayerPtr layer = inputLayers[n];
				NeuralState state = layer->getState();
				WriteNeuralStateToFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "after_" << state.name << ".json", state);
				layer->setState(inStates[n]);
			}
			for (int n = 0; n < (int)outputLayers.size(); n++) {
				NeuralLayerPtr layer = outputLayers[n];
				NeuralState state = layer->getState();
				layer->setState(outStates[n]);
			}

			NeuralState inState;
			for (NeuralLayerPtr layer : outputLayers) {
				if (layer->isLeaf())layer->backpropagate();
			}
			int pad = kernelSize / 2;
			int width = inLayer->width;
			int height = inLayer->height;
			int ow = width - 2 * pad;
			int oh = height - 2 * pad;
			int K = kernelSize*kernelSize;
			int M = (int)outputLayers.size();
			bool hasBias;
			const Knowledge& inResponses = inLayer->responses;
			Knowledge& inResponseChanges = inLayer->responseChanges;
			inResponseChanges.setZero();
			for (NeuralLayerPtr layer : outputLayers) {
				//if (record) {
				//	NeuralState outState = layer->getState();
				//	WriteNeuralStateToFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << outState.name << ".json", outState);
				//}
				const Knowledge& weights = layer->weights;
				const Knowledge& biasWeights = layer->biasWeights;
				Knowledge& outResponses = layer->responses;
				Knowledge& outBiasResponseChanges = layer->biasResponseChanges;
				const Knowledge& outResponseChanges = layer->responseChanges;
				outBiasResponseChanges.setZero();
				//if (record) {
				//	std::cout << "Backprop " << inLayer->getName() << " <- " << layer->getName() << " " << weights.size() << " " << outResponses.size() << " " << biasWeights.size() << std::endl;
				//}
				Knowledge& biasResponses = layer->biasResponses;
				hasBias = (biasWeights.size() > 0);
				for (int j = 0; j < oh; j++) {
					for (int i = 0; i < ow; i++) {
						int idx = i + j*ow;
						float change = outResponseChanges[idx];
						if (hasBias)outBiasResponseChanges[idx] = biasWeights[0] * change*transform.change(biasResponses[idx]);
						for (int jj = 0; jj < kernelSize; jj++) {
							for (int ii = 0; ii < kernelSize; ii++) {
								int shifted = (i + ii) + width*(j + jj);
								inResponseChanges[shifted] += 1.0f;// weights[ii + kernelSize*jj] * change;
							}
						}
					}
				}
			}
			for (int k = 0; k < 20; k++) {
				std::cout << "Out response " << inResponseChanges[500+k] <<" "<<kernelSize<<" "<<ow<<" "<<oh<< std::endl;
			}
			
#pragma omp parallel for
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					int idx = i + j*width;
					inResponseChanges[idx] *= transform.change(inResponses[idx]) / M;
				}
			}
			
			for (NeuralLayerPtr layer : outputLayers) {
				Knowledge& weightChanges = layer->weightChanges;
				Knowledge& biasWeightChanges = layer->biasWeightChanges;
				Knowledge& outBiasResponseChanges = layer->biasResponseChanges;
				Knowledge& biasResponses = layer->biasResponses;
#pragma omp parallel for
				for (int j = 0; j < oh; j++) {
					for (int i = 0; i < ow; i++) {
						int idx = i + ow*j;
						float inResponse = inResponses[idx];
						if (hasBias)biasWeightChanges[idx] += biasResponses[idx] * outBiasResponseChanges[idx];
						for (int jj = 0; jj < kernelSize; jj++) {
							for (int ii = 0; ii < kernelSize; ii++) {
								weightChanges[ii + kernelSize*jj] += inResponse*inResponseChanges[(i + ii) + width*(j + jj)];
							}
						}
					}
				}
			}
			if (record) {
				for (int n = 0; n < (int)inputLayers.size(); n++) {
					NeuralLayerPtr layer = inputLayers[n];
					NeuralState state = layer->getState();
					WriteNeuralStateToFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "opt_" << state.name << ".json", state);
				}
			}
		} else {
			NeuralFilter::backpropagate();
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