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
#include <NeuralSignal.h>
#include "NeuralLayerRegion.h"
#include "NeuralKnowledge.h"
#include <vector>
#include <set>
namespace tiny_dnn{
	class Device;
}
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
	int id;
	std::string name;
	bool trainable;
	bool visited;
	bool initialized;
	bool parallelize;
	BackendType backendType;
	NeuralSystem* sys;
	aly::NeuralLayerRegionPtr layerRegion;
	aly::GraphDataPtr graph;
	std::function<void(Storage& data, int fanIn, int fanOut)> weightInitFunc;
	std::function<void(Storage& data, int fanIn, int fanOut)> biasInitFunc;
	Storage weightDifference;
	tiny_dnn::Device* device_ptr_=nullptr;
public:
	friend void Connect(const std::shared_ptr<NeuralLayer>& head,
			const std::shared_ptr<NeuralLayer>& tail, int head_index,
			int tail_index);
	int inputChannels;
	int outputChannels;
	virtual std::vector<aly::dim3> getInputDimensions() const = 0;
	virtual std::vector<aly::dim3> getOutputDimensions() const = 0;
	virtual void setInputShape(const aly::int3& in_shape) {
		throw std::runtime_error(
				"Can't set shape. Shape inferring not applicable for this "
						"layer (yet).");
	};

	tiny_dnn::Device* device() const { return device_ptr_; }
	void clearGradients();
	aly::int3 getOutputDimensions(size_t idx) const {
		return getOutputDimensions()[idx];
	}
	aly::int3 getInputDimensions(size_t idx) const {
		return getInputDimensions()[idx];
	}
	size_t getOutputDimensionSize() const {
		return getOutputDimensions().size();
	}
	size_t getInputDimensionSize() const {
		return getInputDimensions().size();
	}
	float getAspect() const {
		return 1.0f;
	}
	SignalPtr getInput(size_t i) {
		if (inputs[i].get() == nullptr) {
			inputs[i] = SignalPtr(
					new NeuralSignal(nullptr, getInputDimensions(i), inputTypes[i]));
		}
		return inputs[i];
	}
	SignalPtr getOutput(size_t i) {
		if (outputs[i].get() == nullptr) {
			outputs[i] = SignalPtr(
					new NeuralSignal(this, getOutputDimensions(i), outputTypes[i]));
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
	void setParallelize(bool parallelize) {
		this->parallelize = parallelize;
	}
	void setBackendType(BackendType backend_type) {
		backendType = backend_type;
	}
	BackendType getBackendType() const {
		return backendType;
	}
	std::vector<const Storage*> getInputWeights() const;
	std::vector<const Storage*> getOutputWeights() const;
	std::vector<const Storage*> getInputGradient() const;
	std::vector<const Storage*> getOutputGradient() const;
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) = 0;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad) = 0;
	virtual void setSampleCount(size_t sample_count);

	void updateWeights(
			const std::function<void(Storage& dW, Storage& W, bool parallel)>& optimizer,
			int batch_size);
	bool hasSameWeights(const NeuralLayer &rhs, float_t eps) const;
	void initializeWeights();
	void setup(bool reset_weight);
	void setOutputGradients(
			const std::vector<std::vector<const Storage*>>& grad);
	void setInputData(const std::vector<std::vector<const Storage*>>& data);
	void getOutput(std::vector<Tensor*>& out) const;
	void getOutput(std::vector<const Tensor*>& out) const;
	std::vector<std::shared_ptr<NeuralLayer>> getOutputLayers() const;
	std::vector<NeuralLayer*> getInputLayers() const;
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
		return (outputs.size() != 0);
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

	void setSystem(NeuralSystem* s) {
		sys = s;
	}
	void setRegionDirty(bool d);
	aly::NeuralLayerRegionPtr getRegion();
	bool hasRegion() const {
		return (layerRegion.get() != nullptr && layerRegion->parent != nullptr);
	}
	bool isVisible() const;
	std::vector<std::shared_ptr<NeuralSignal>>& getInputSignals() {
		return inputs;
	}
	const std::vector<std::shared_ptr<NeuralSignal>>& getInputSignals() const {
		return inputs;
	}
	std::vector<std::shared_ptr<NeuralSignal>>& getOutputSignals() {
		return outputs;
	}
	const std::vector<std::shared_ptr<NeuralSignal>>& getOutputSignals() const {
		return outputs;
	}

	bool isRoot() const {
		return (inputs.size() == 0);
	}
	bool isLeaf() const {
		return (outputs.size() == 0);
	}
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
	virtual ~NeuralLayer() {
	}
};

typedef std::shared_ptr<NeuralLayer> NeuralLayerPtr;
void Connect(const std::shared_ptr<NeuralLayer>& head, const std::shared_ptr<NeuralLayer>& tail,
		int head_index = 0, int tail_index = 0);
}
#endif
