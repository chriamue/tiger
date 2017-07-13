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

#include "ActivationLayer.h"
#include "AddElementsLayer.h"
#include "AveragePoolingLayer.h"
#include "AverageUnpoolingLayer.h"
#include "BatchNormalizationLayer.h"
#include "ConcatLayer.h"
#include "DeconvolutionLayer.h"
#include "FullyConnectedLayer.h"
#include "InputLayer.h"
#include "LinearLayer.h"
#include "MaxPoolingLayer.h"
#include "PartialConnectedLayer.h"
#include "TanhLayer.h"
#include "ConvolutionLayer.h"
#include "NeuralLossFunction.h"
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
	std::vector<NeuralLayerPtr> inputLayers;
	std::vector<NeuralLayerPtr> outputLayers;
	NeuralKnowledge knowledge;
	std::string name;
	void reorderForLayerwiseProcessing(const std::vector<Tensor> &input,
			std::vector<std::vector<const Storage *>> &output);

public:
	typedef std::vector<NeuralLayerPtr>::iterator iterator;
	typedef std::vector<NeuralLayerPtr>::const_iterator const_iterator;
	void bprop(const NeuralLossFunction& loss, const std::vector<Storage> &out,
			const std::vector<Storage> &t, const std::vector<Storage> &t_cost);
	void bprop(const NeuralLossFunction& loss, const std::vector<Tensor> &out,
			const std::vector<Tensor> &t, const std::vector<Tensor> &t_cost);
	Storage fprop(const Storage &in);
	std::vector<Storage> fprop(const std::vector<Storage> &in);
	std::vector<Tensor> fprop(const std::vector<Tensor> &in);
	size_t size() const {
		return layers.size();
	}
	iterator begin() {
		return layers.begin();
	}
	iterator end() {
		return layers.end();
	}
	const_iterator begin() const {
		return layers.begin();
	}
	const_iterator end() const {
		return layers.end();
	}
	NeuralLayerPtr operator[](size_t index) {
		return layers[index];
	}
	const NeuralLayerPtr operator[](size_t index) const {
		return layers[index];
	}
	float getTargetValueMin() const;
	float getTargetValueMax() const;
	size_t getInputDataSize() const;
	size_t getOutputDataSize() const;
	std::vector<Tensor> mergeOutputs();
	void setKnowledge(const NeuralKnowledge& k);
	double accumulate(const NeuralLayerPtr& layer, const aly::Image1f& output);
	double accumulate(const NeuralLayerPtr& layer,
			const std::vector<float>& output);
	void reset();

	NeuralKnowledge& getKnowledge() {
		return knowledge;
	}
	NeuralKnowledge& updateKnowledge();
	const NeuralKnowledge& getKnowledge() const {
		return knowledge;
	}
	void updateWeights(NeuralOptimizer& optimizer,int batch_size);
	void initialize();
	void setPhase(NetPhase phase);
	void normalize(const std::vector<Tensor> &inputs,
			std::vector<Tensor> &normalized);
	void normalize(const std::vector<Storage> &inputs,
			std::vector<Tensor> &normalized);
	void normalize(const std::vector<int> &inputs,
			std::vector<Tensor> &normalized);
	float getLoss(const NeuralLossFunction& loss, const std::vector<Tensor> &in,
			const std::vector<Tensor> &t);
	float getLoss(const NeuralLossFunction& loss, const std::vector<Storage> &in,
			const std::vector<Storage> &t);
	float getLoss(const NeuralLossFunction& loss, const std::vector<Storage> &in,
			const std::vector<Tensor> &t);
	float getLoss(const NeuralLossFunction& loss, const std::vector<int> &in,
			const std::vector<Tensor> &t);
	bool gradientCheck(const NeuralLossFunction& func, const std::vector<Tensor> &in,
			const std::vector<std::vector<int>> &t, float eps,
			GradientCheck mode);
	void label2vec(const int *t, int num, std::vector<Storage> &vec) const;
	void label2vec(const std::vector<int> &labels,
			std::vector<Storage> &vec) const;
	Storage predict(const Storage &in);
	Tensor predict(const Tensor &in);
	bool calculateDelta(const NeuralLossFunction& loss, const std::vector<Tensor> &in,
			const std::vector<Tensor> &v, Storage &w, Tensor &dw,
			int check_index, double eps);
	std::vector<Tensor> predict(const std::vector<Tensor>& in);
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
	const std::vector<NeuralLayerPtr>& getInputLayers() const {
		return inputLayers;
	}
	std::vector<NeuralLayerPtr>& getInputLayers() {
		return inputLayers;
	}
	const std::vector<NeuralLayerPtr>& getOutputLayers() const {
		return outputLayers;
	}
	std::vector<NeuralLayerPtr>& getOutputLayers() {
		return outputLayers;
	}
	std::vector<Storage> test(const std::vector<Storage> &in);
	NeuralSystem(const std::string& name,const std::shared_ptr<aly::NeuralFlowPane>& pane);
	std::vector<Tensor> forward(const std::vector<Tensor> &in_data);
	void evaluate();
	void setup(bool reset_weight);
	void clearGradients();
	void backward(const std::vector<Tensor> &out_grad);
	void build(const std::vector<NeuralLayerPtr>& input,
			const std::vector<NeuralLayerPtr> &output);
	void build(NeuralLayerPtr input, NeuralLayerPtr output) {
		build(std::vector<NeuralLayerPtr> { input },
				std::vector<NeuralLayerPtr> { output });
	}
	void build(const std::initializer_list<NeuralLayerPtr>& input,
			const std::initializer_list<NeuralLayerPtr> &output) {
		build(std::vector<NeuralLayerPtr>(input.begin(), input.end()),
				std::vector<NeuralLayerPtr>(output.begin(), output.end()));
	}
};
typedef std::shared_ptr<NeuralSystem> NeuralSystemPtr;

}
#endif
