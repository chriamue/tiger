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
#ifndef NeuralLayer_H_
#define NeuralLayer_H_
#include <AlloyMath.h>
#include <AlloyContext.h>
#include <AlloyExpandTree.h>
#include <AlloyWidget.h>
#include <AlloyGraphPane.h>
#include <Signal.h>
#include "NeuralLayerRegion.h"
#include "NeuralOptimization.h"
#include "NeuralKnowledge.h"
#include <vector>
#include <set>

class TigerApp;
namespace aly {
class NeuralFlowPane;
}
namespace tgr {
std::string MakeID(int len = 8);
class NeuralSystem;
struct NeuralState {
	std::string name;
	Knowledge weights;
	Knowledge weightChanges;
	Knowledge biasWeights;
	Knowledge biasWeightChanges;
	Knowledge responses;
	Knowledge responseChanges;
	Knowledge biasResponses;
	Knowledge biasResponseChanges;
	template<class Archive> void save(Archive & ar) const {
		ar(CEREAL_NVP(name), CEREAL_NVP(weights), CEREAL_NVP(weightChanges),
				CEREAL_NVP(biasWeights), CEREAL_NVP(biasWeightChanges),
				CEREAL_NVP(responses), CEREAL_NVP(responseChanges),
				CEREAL_NVP(biasResponses), CEREAL_NVP(biasResponseChanges));
	}
	template<class Archive> void load(Archive & ar) {
		ar(CEREAL_NVP(name), CEREAL_NVP(weights), CEREAL_NVP(weightChanges),
				CEREAL_NVP(biasWeights), CEREAL_NVP(biasWeightChanges),
				CEREAL_NVP(responses), CEREAL_NVP(responseChanges),
				CEREAL_NVP(biasResponses), CEREAL_NVP(biasResponseChanges));
	}
};
void WriteNeuralStateToFile(const std::string& file, const NeuralState& params);
void ReadNeuralStateFromFile(const std::string& file, NeuralState& params);
class NeuralLayer {
protected:
	std::vector<SignalPtr> inputs;
	std::vector<SignalPtr> outputs;
	std::vector<ChannelType> inputTypes;
	std::vector<ChannelType> outputTypes;
	std::vector<NeuralLayer*> children;
	std::vector<NeuralLayer*> dependencies;
	std::shared_ptr<NeuralOptimization> optimizer;
	std::string name;
	bool trainable;
	bool visited;
	bool initialized;
	aly::NeuralLayerRegionPtr layerRegion;
	aly::GraphDataPtr graph;
	NeuralSystem* sys;
	int id;
	std::function<void(Storage& data, int fanIn, int fanOut)> weightInitFunc;
	std::function<void(Storage& data, int fanIn, int fanOut)> biasInitFunc;
	std::vector<Tensor *> fowardInData;
	std::vector<Tensor *> fowardInGradient;
	std::vector<Tensor *> backwardInData;
	std::vector<Tensor *> backwardInGradient;
	std::vector<Tensor *> backwardOutData;
	std::vector<Tensor *> backwardOutGradient;
	Storage weightDifference;
public:
	int inputChannels;
	int outputChannels;
	virtual std::vector<int3> getInputDimensions() const = 0;
	virtual std::vector<int3> getOutputDimensions() const = 0;
	void clearGradients();
	int3 getOutputDimensions(size_t idx) const {
		return getOutputDimensions()[idx];
	}
	int3 getInputDimensions(size_t idx) const {
		return getInputDimensions()[idx];
	}
	size_t getOutputDimensionSize() const {
		return getOutputDimensions().size();
	}
	size_t getInputDimensionSize() const {
		return getInputDimensions().size();
	}
	SignalPtr getInput(size_t i) {
		if (inputs[i].get() == nullptr) {
			inputs[i] = SignalPtr(
					new Signal(nullptr, getInputDimensions(i), inputTypes[i]));
		}
		return inputs[i];
	}
	SignalPtr getOutput(size_t i) {
		if (outputs[i].get() == nullptr) {
			outputs[i] = SignalPtr(
					new Signal(this, getOutputDimensions(i), outputTypes[i]));
		}
		return outputs[i];
	}
	SignalPtr getInput(size_t i) const {
		return inputs[i];
	}
	SignalPtr getOutput(size_t i) const {
		return outputs[i];
	}
	Storage& getInputWeights(size_t idx) {
		return getInput(idx)->weight.front();
	}
	Storage& getOutputWeights(size_t idx) {
		return getOutput(idx)->weight.front();
	}
	const Storage& getInputWeights(size_t idx) const {
		return getInput(idx)->weight.front();
	}
	const Storage& getOutputWeights(size_t idx) const {
		return getOutput(idx)->weight.front();
	}
	std::vector<const Storage*> getInputWeights() const;
	std::vector<const Storage*> getOutputWeights() const;
	std::vector<const Storage*> getInputGradient() const;
	std::vector<const Storage*> getOutputGradient() const;
	void updateWeights(const std::function<void(Storage& dW,Storage& W,bool parallel)>& optimizer, int batch_size);
	bool hasSameWeights(const NeuralLayer &rhs, float_t eps) const;
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) = 0;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad) = 0;
	virtual void setSampleCount(size_t sample_count);
	void initializeWeights();
	void setup(bool reset_weight);
	void setOutputGradients(
			const std::vector<std::vector<const Storage*>>& grad);
	void setInputData(const std::vector<std::vector<const Storage*>>& data);
	void getOutput(std::vector<Tensor*>& out) const;
	void forward(const std::vector<Tensor>&input, std::vector<Tensor*>& out);
	std::vector<Tensor> backward(const std::vector<Tensor>& out_grads);
	void forward();
	void backward();
	virtual void post() {
	}
	virtual int getFanInSize() const {
		return getInputDimensions()[0].x;
	}
	virtual int getFanOutSize() const {
		return getOutputDimensions()[0].x;
	}

	void setWeightInitialization(
			const std::function<void(Storage& data, int fanIn, int fanOut)>& func) {
		weightInitFunc = func;
	}
	void setBiasInitialization(
			const std::function<void(Storage& data, int fanIn, int fanOut)>& func) {
		biasInitFunc = func;
	}
	std::vector<ChannelType> getInputTypes() const {
		return inputTypes;
	}
	std::vector<ChannelType> getOutputTypes() const {
		return outputTypes;
	}
	int getId() const {
		return id;
	}
	void setId(int id);
	std::shared_ptr<aly::NeuralFlowPane> getFlow() const;
	aly::GraphDataPtr getGraph() const {
		return graph;
	}
	void expand();
	double accumulate(double r);
	bool hasChildren() const {
		return (children.size() != 0);
	}
	bool isVisited() const {
		return visited;
	}
	bool isTrainable() const {
		return trainable;
	}
	void setTrainable(bool t) {
		trainable = t;
	}
	void setVisited(bool v) {
		visited = v;
	}

	void setOptimizer(const std::shared_ptr<NeuralOptimization>& opt) {
		optimizer = opt;
	}
	void setSystem(NeuralSystem* s) {
		sys = s;
	}
	void setRegionDirty(bool d);
	aly::NeuralLayerRegionPtr getRegion();
	bool hasRegion() const {
		return (layerRegion.get() != nullptr && layerRegion->parent != nullptr);
	}
	bool isVisible() const;
	std::vector<std::shared_ptr<Signal>>& getInputSignals() {
		return inputs;
	}
	const std::vector<std::shared_ptr<Signal>>& getInputSignals() const {
		return inputs;
	}
	std::vector<std::shared_ptr<Signal>>& getOutputSignals() {
		return outputs;
	}
	const std::vector<std::shared_ptr<Signal>>& getOutputSignals() const {
		return outputs;
	}
	std::vector<NeuralLayer*>& getChildren() {
		return children;
	}
	std::vector<NeuralLayer*>& getDependencies() {
		return dependencies;
	}
	bool visitedChildren() const;
	bool visitedDependencies() const;

	const std::vector<NeuralLayer*>& getChildren() const {
		return children;
	}
	const std::vector<NeuralLayer*>& getDependencies() const {
		return dependencies;
	}
	bool isRoot() const {
		return (dependencies.size() == 0);
	}
	bool isLeaf() const {
		return (children.size() == 0);
	}
	void addChild(const std::shared_ptr<NeuralLayer>& layer);
	void setName(const std::string& n) {
		name = n;
	}
	std::string getName() const {
		return name;
	}
	void initialize(const aly::ExpandTreePtr& tree,
			const aly::TreeItemPtr& treeItem);
	NeuralLayer(const std::string& name,
			const std::vector<ChannelType>& inTypes,
			const std::vector<ChannelType>& outTypes);
};
typedef std::shared_ptr<NeuralLayer> NeuralLayerPtr;
}
#endif
