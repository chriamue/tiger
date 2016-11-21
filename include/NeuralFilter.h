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
#ifndef NEURALFILTER_H_
#define NEURALFILTER_H_
#include "NeuralLayer.h"
#include "NeuralSystem.h"
namespace tgr {
	class NeuralFilter {
		protected:
			NeuralLayerPtr inputLayer;
			std::vector<NeuralLayerPtr> outputLayers;
		public:
			NeuralLayerPtr& getInputLayer() {
				return inputLayer;
			}
			const NeuralLayerPtr& getInputLayer() const {
				return inputLayer;
			}
			std::vector<NeuralLayerPtr>& getOutputLayers() {
				return outputLayers;
			}

			const std::vector<NeuralLayerPtr>& getOutputLayers() const {
				return outputLayers;
			}
			NeuralLayerPtr& getOutputLayer(size_t idx) {
				return outputLayers[idx];
			}
			const NeuralLayerPtr& getOutputLayer(size_t idx) const {
				return outputLayers[idx];
			}

			size_t getOutputSize() const {
				return outputLayers.size();
			}
			NeuralFilter() {}
			virtual ~NeuralFilter() {}
			virtual void attach(NeuralSystem& sys) = 0;
	};
	class ConvolutionFilter:public NeuralFilter {
	protected:
		int kernelX, kernelY;
	public:
		ConvolutionFilter(int width, int height, int kernelX, int kernelY, int features);
		ConvolutionFilter(const NeuralLayerPtr& inputLayer, int kernelX, int kernelY, int features);

		virtual void attach(NeuralSystem& sys) override;
	};
	typedef std::shared_ptr<NeuralFilter> NeuralFilterPtr;
	typedef std::shared_ptr<ConvolutionFilter> ConvolutionFilterPtr;
}
#endif