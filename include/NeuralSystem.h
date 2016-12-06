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
#include "NeuralKnowledge.h"
#include <map>
namespace aly {
	class NeuralFlowPane;
}
namespace tgr {
	class NeuralLayer;
	class NeuralFilter;
	class NeuralSystem {
	protected:
		std::vector<NeuralLayerPtr> layers;
		std::vector<std::shared_ptr<NeuralFilter>> filters;
		std::vector<NeuralLayerPtr> roots;
		std::vector<NeuralLayerPtr> leafs;
		bool initialized;
		std::shared_ptr<aly::NeuralFlowPane> flowPane;
		NeuralLayerPtr inputLayer, outputLayer;
		NeuralKnowledge knowledge;
	public:

		void evaluate();
		void backpropagate();
		bool optimize();
		void setKnowledge(const NeuralKnowledge& k);
		void setInput(const NeuralLayerPtr& layer) {
			inputLayer = layer;
		}
		void setOutput(const NeuralLayerPtr& layer) {
			outputLayer = layer;
		}
		NeuralLayerPtr getInput() const {
			return inputLayer;
		}
		NeuralLayerPtr getOutput() const {
			return outputLayer;
		}
		void setOptimizer(const NeuralOptimizationPtr& opt);
		double accumulate(const NeuralLayerPtr& layer, const aly::Image1f& output);
		double accumulate(const NeuralLayerPtr& layer, const std::vector<float>& output);

		void reset();
		inline double accumulate(const aly::Image1f& output) {
			return accumulate(outputLayer, output);
		}
		inline double accumulate(const std::vector<float>& output) {
			return accumulate(outputLayer, output);
		}
		NeuralKnowledge& getKnowledge() {
			return knowledge;
		}
		NeuralKnowledge& updateKnowledge();
		const NeuralKnowledge& getKnowledge() const {
			return knowledge;
		}
		void initialize();
		std::shared_ptr<aly::NeuralFlowPane> getFlow() const {
			return flowPane;
		}
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
		NeuralSystem(const std::shared_ptr<aly::NeuralFlowPane>& pane);
		void setLayer(const NeuralLayerPtr& layer, const aly::Image1f& input);
		void setLayer(const NeuralLayerPtr& layer, const std::vector<float>& input);
		void getLayer(const NeuralLayerPtr& layer, aly::Image1f& input);
		void getLayer(const NeuralLayerPtr& layer, std::vector<float>& input);

		inline void setInput(const aly::Image1f& input) {
			setLayer(inputLayer, input);
		}
		inline void setInput(const std::vector<float>& input) {
			setLayer(inputLayer, input);
		}
		inline void getOutput(aly::Image1f& out) {
			getLayer(outputLayer, out);
		}
		inline void getOutput(std::vector<float>& out) {
			getLayer(outputLayer, out);
		}
		void initializeWeights(float minW = 0.0f, float maxW = 1.0f);
		void add(const std::shared_ptr<NeuralFilter>& filter, const NeuronFunction& func = Tanh());
	};
	typedef std::shared_ptr<NeuralSystem> NeuralSystemPtr;

}
#endif