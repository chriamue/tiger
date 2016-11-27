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
#ifndef _NEURAL_SYSTEM_H_
#define _NEURAL_SYSTEM_H_
#include "NeuralLayer.h"
#include "AlloyExpandTree.h"
#include <map>
namespace tgr {
	class NeuralFilter;
	class NeuralSystem {
	protected:
		std::vector<NeuralLayerPtr> layers;
		std::vector<NeuralLayer*> backpropLayers;
		std::vector<NeuralLayerPtr> roots;
		std::vector<NeuralLayerPtr> leafs;
		std::map<Terminal,float> input;
		std::map<Terminal,float> output;
	public:
		void evaluate();
		void backpropagate();
		void pushInput();
		void pullOutput();
		void accumulateError();
		void resetError();
		void initialize();
		void train(float learningRate);
		Neuron* getNeuron(const Terminal& t) const;
		void initialize(const aly::ExpandTreePtr& tree);
		const std::vector<NeuralLayerPtr>& getRoots() const {
			return roots;
		}
		std::vector<NeuralLayerPtr>& getRoots() {
			return roots;
		}
		const std::vector<NeuralLayerPtr>& getLayers() const {
			return layers;
		}
		std::vector<NeuralLayerPtr>& getLayers() {
			return layers;
		}
		void setInput(const Terminal& t, float value);
		float getOutput(const Terminal& t);
		Terminal addInput(int i,int j,const NeuralLayerPtr& layer, float value=0.0f);
		Terminal addOutput(int i, int j, const NeuralLayerPtr& layer, float value=0.0f);

		SignalPtr add(Terminal source,Terminal target,float weight=0.0f);
		void add(const std::shared_ptr<NeuralFilter>& filter);
	
	};
}
#endif