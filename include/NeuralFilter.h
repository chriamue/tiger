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
class TigerApp;
namespace tgr {
	class NeuralFilter {
		protected:
			std::vector<NeuralLayerPtr> inputLayers;
			std::vector<NeuralLayerPtr> outputLayers;
			TigerApp* app;
		public:
			virtual bool isTrainable() const {
				return true;
			}
			std::vector<NeuralLayerPtr>& getInputLayers() {
				return inputLayers;
			}
			const std::vector<NeuralLayerPtr>& getInputLayers() const {
				return inputLayers;
			}
			NeuralLayerPtr& getInputLayer(size_t idx) {
				return inputLayers[idx];
			}
			const NeuralLayerPtr& getInputLayer(size_t idx) const {
				return inputLayers[idx];
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
			size_t getInputSize() const {
				return inputLayers.size();
			}
			NeuralFilter(TigerApp* app):app(app) {}
			virtual ~NeuralFilter() {}
			virtual void initialize(NeuralSystem& sys) = 0;
	};

	typedef std::shared_ptr<NeuralFilter> NeuralFilterPtr;
}
#endif