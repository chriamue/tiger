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
#include "NeuralLayer.h"
#include "AlloyUnits.h"
#include "AlloyDrawUtil.h"
#include "TigerApp.h"
#include "NeuralFlowPane.h"
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
using namespace aly;
namespace tgr {
std::string MakeID(int len) {
	std::stringstream ss;
	static const char lookUp[33] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345";
	for (int i = 0; i < len; i++) {
		ss << lookUp[RandomUniform(0, 31)];
	}
	return ss.str();
}
void WriteNeuralStateToFile(const std::string& file,
		const NeuralState& params) {
	std::string ext = GetFileExtension(file);
	if (ext == "json") {
		std::ofstream os(file);
		cereal::JSONOutputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	} else if (ext == "xml") {
		std::ofstream os(file);
		cereal::XMLOutputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	} else {
		std::ofstream os(file, std::ios::binary);
		cereal::PortableBinaryOutputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	}
}
void ReadNeuralStateFromFile(const std::string& file, NeuralState& params) {
	std::string ext = GetFileExtension(file);
	if (ext == "json") {
		std::ifstream os(file);
		cereal::JSONInputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	} else if (ext == "xml") {
		std::ifstream os(file);
		cereal::XMLInputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	} else {
		std::ifstream os(file, std::ios::binary);
		cereal::PortableBinaryInputArchive archive(os);
		archive(cereal::make_nvp("neuralstate", params));
	}
}
void Connect(const NeuralLayerPtr& head, const NeuralLayerPtr& tail,
		int head_index, int tail_index) {
	auto out_shape = head->getOutputDimensions(head_index);
	auto in_shape = tail->getInputDimensions(tail_index);
	head->setup(false);
	// todo (karandesai) enable shape inferring for all layers
	// currently only possible for activation layers.
	if (in_shape.size() == 0) {
		tail->setInputShape(out_shape);
		in_shape = out_shape;
	}
	if (out_shape.size() != in_shape.size()) {
		throw std::runtime_error("Layer Dimension mismatch");
	}
	if (head->outputs[head_index].get() == nullptr) {
		throw std::runtime_error("output edge must not be null");
	}
	tail->inputs[tail_index] = head->outputs[head_index];
	tail->inputs[tail_index]->addOutput(tail);
}
NeuralLayer::NeuralLayer(const std::string& name,
		const std::vector<ChannelType>& inTypes,
		const std::vector<ChannelType>& outTypes) :
		id(-1), inputSize(-1, -1, -1), outputSize(-1, -1, -1), name(name), inputTypes(
				inTypes), outputTypes(outTypes) {
	inputChannels = (int) inputTypes.size();
	outputChannels = (int) outputTypes.size();
	inputs.resize(inputChannels);
	outputs.resize(outputChannels);
	initialized = false;
	trainable = true;
	visited = false;
	parallelize = false;
	sys = nullptr;
	weightInitFunc=[this](Storage& data, int fanIn, int fanOut)  {
		float weight_base = std::sqrt(6.0f / (fanIn + fanOut));
		for(float& val:data){
			val=RandomUniform(-weight_base,weight_base);
		}
	};
	biasInitFunc=[this](Storage& data, int fanIn, int fanOut)  {
		for(float& val:data){
			val=0.0f;
		}
	};
	backendType = BackendType::internal;
}
void NeuralLayer::setId(int i) {
	id = i;
}
std::shared_ptr<aly::NeuralFlowPane> NeuralLayer::getFlow() const {
	return sys->getFlow();
}
void NeuralLayer::setRegionDirty(bool b) {
	if (layerRegion.get() != nullptr) {
		layerRegion->setDirty(b);
	}
}

bool NeuralLayer::isVisible() const {
	if (layerRegion.get() != nullptr && layerRegion->parent != nullptr) {
		return layerRegion->isVisible();
	} else {
		return false;
	}
}
aly::NeuralLayerRegionPtr NeuralLayer::getRegion() {
	if (layerRegion.get() == nullptr) {
		aly::dim3 outDims = getOutputSize();

		float2 dims = float2(240.0f * outDims.z,
				240.0f * outDims.y / outDims.x);
		if (dims.x > 2048.0f) {
			dims /= 2048.0f;
		}
		dims += NeuralLayerRegion::getPadding();

		layerRegion = NeuralLayerRegionPtr(
				new NeuralLayerRegion(name, this,
						CoordPerPX(0.5f, 0.5f, -dims.x * 0.5f, -dims.y * 0.5f),
						CoordPX(dims.x, dims.y)));
		if (hasChildren()) {
			layerRegion->setExpandable(true);
			for (auto child : getOutputLayers()) {
				child->getRegion();
			}
		}
		layerRegion->onHide = [this]() {
			sys->getFlow()->update();
		};
		layerRegion->onExpand = [this]() {
			expand();
		};
	}
	return layerRegion;
}
template<typename Result, typename T, typename Pred>
std::vector<Result> map_(const std::vector<T> &vec, Pred p) {
	std::vector<Result> res(vec.size());
	for (size_t i = 0; i < vec.size(); ++i) {
		res[i] = p(vec[i]);
	}
	return res;
}
void NeuralLayer::setSampleCount(size_t sample_count) {
	// increase the size if necessary - but do not decrease
	auto resize = [sample_count](Tensor*tensor) {
		tensor->resize(sample_count,(*tensor)[0]);
	};
	for (size_t i = 0; i < inputChannels; i++) {
		if (!isTrainableWeight(inputTypes[i])) {
			resize(&getInput(i)->value);
		}
		resize(&getInput(i)->change);
	}

	for (int i = 0; i < outputChannels; i++) {
		if (!isTrainableWeight(outputTypes[i])) {
			resize(&getOutput(i)->value);
		}
		resize(&getOutput(i)->change);
	}
}
void NeuralLayer::forward() {
	// the computational graph
	fowardInData.resize(inputChannels);
	fowardInGradient.resize(outputChannels);
	// Organize input/output vectors from storage (computational graph).
	// Internally ith_in_node() will create a connection/edge in the
	// computational graph and will allocate memory in case that it's not
	// done yet.
	for (int i = 0; i < inputChannels; i++) {
		fowardInData[i] = &getInput(i)->value;
	}
	// resize outs and stuff to have room for every input sample in
	// the batch
	setSampleCount(fowardInData[0]->size());

	// Internally ith_out_node() will create a connection/edge to the
	// computational graph and will allocate memory in case that it's not
	// done yet. In addition, gradient vector are initialized to default
	// values.
	for (int i = 0; i < outputChannels; i++) {
		fowardInGradient[i] = &getOutput(i)->value;
		getOutput(i)->clearGradients();
	}
	// call the forward computation kernel/routine
	forwardPropagation(fowardInData, fowardInGradient);
	setRegionDirty(true);
}

void NeuralLayer::backward() {
	backwardInData.resize(inputChannels);
	backwardInGradient.resize(inputChannels);
	backwardOutData.resize(outputChannels);
	backwardOutGradient.resize(outputChannels);
	// organize input/output vectors from storage
	for (int i = 0; i < inputChannels; i++) {
		SignalPtr nd = getInput(i);
		backwardInData[i] = &nd->value;
		backwardInGradient[i] = &nd->change;
	}
	for (int i = 0; i < outputChannels; i++) {
		SignalPtr nd = getOutput(i);
		backwardOutData[i] = &nd->value;
		backwardOutGradient[i] = &nd->change;
	}
	backwardPropagation(backwardInData, backwardOutData, backwardOutGradient,
			backwardInGradient);
}
std::vector<Tensor> NeuralLayer::backward(
		const std::vector<Tensor>& out_grads) { // for test
	setup(false);
	std::vector<std::vector<const Storage*>> grads2;
	grads2.resize(out_grads.size());
	for (size_t i = 0; i < out_grads.size(); ++i) {
		grads2[i].resize(out_grads[i].size());
		for (size_t j = 0; j < out_grads[i].size(); ++j) {
			grads2[i][j] = &(out_grads[i][j]);
		}
	}
	setOutputGradients(grads2);
	backward();
	return map_<Tensor>(inputs, [](SignalPtr e) {return e->change;});
}
void NeuralLayer::forward(const std::vector<Tensor>& input,
		std::vector<Tensor*>& out) {  // for test
	// allocate data in the computational graph without
	// resetting the weights.
	setup(false);
	std::vector<std::vector<const Storage*>> input2;
	input2.resize(input.size());
	for (size_t i = 0; i < input.size(); ++i) {
		input2[i].resize(input[i].size());
		for (size_t j = 0; j < input[i].size(); ++j) {
			input2[i][j] = &(input[i][j]);
		}
	}
	// the incoming data is forwarded to the computational graph.
	setInputData(input2);
	// pick up the data from the computational graph and perform
	// computation.
	forward();
	// retrieve computed data and return values in form of 4D tensor.
	getOutput(out);
}

void NeuralLayer::getOutput(std::vector<Tensor*>& out) const {
	out.clear();
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data) {
			out.push_back(&(getOutput(i)->value));
		}
	}
}
void NeuralLayer::getOutput(std::vector<const Tensor*>& out) const {
	out.clear();
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data) {
			out.push_back(&(getOutput(i)->value));
		}
	}
}
std::vector<NeuralLayerPtr> NeuralLayer::getOutputLayers() const {
	std::vector<NeuralLayerPtr> vecs;
	for (SignalPtr e : outputs) {
		if (e.get() != nullptr) {
			const std::vector<NeuralLayerPtr>& n = e->outputs;
			vecs.insert(vecs.end(), n.begin(), n.end());
		}
	}
	return vecs;
}
std::vector<NeuralLayer*> NeuralLayer::getInputLayers() const {
	std::vector<NeuralLayer*> vecs;
	for (SignalPtr e : inputs) {
		if (e.get() != nullptr&&e->input!=nullptr) {
			vecs.push_back(e->input);
		}
	}
	return vecs;
}

void NeuralLayer::setOutputGradients(
		const std::vector<std::vector<const Storage*>>& grad) {
	size_t n = 0;
	size_t cnt = grad.size();
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] != ChannelType::data)
			continue;
		Tensor& dst_grad = getOutput(i)->change;
		assert(n < cnt);
		const std::vector<const Storage*>& storage = grad[n++];
		size_t sz = storage.size();
		dst_grad.resize(sz);
		for (size_t j = 0; j < sz; ++j) {
			dst_grad[j] = *storage[j];
		}
	}
}
void NeuralLayer::setOutputData(const Tensor& data) {
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data){
			getOutput(i)->value=data;
			break;
		}
	}
}
void NeuralLayer::setOutputData(const aly::Image1f& data) {
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data){
			getOutput(i)->setValue(data);
			break;
		}
	}
}
void NeuralLayer::setOutputData(const aly::Image3f& data) {
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data){
			getOutput(i)->setValue(data);
			break;
		}
	}
}
void NeuralLayer::setOutputData(const aly::Image4f& data) {
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data){
			getOutput(i)->setValue(data);
			break;
		}
	}
}

void NeuralLayer::setInputData(const aly::Image1f& data) {
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] == ChannelType::data){
			getInput(i)->setValue(data);
			break;
		}
	}
}
void NeuralLayer::setInputData(const aly::Image3f& data) {
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] == ChannelType::data){
			getInput(i)->setValue(data);
			break;
		}
	}
}
void NeuralLayer::setInputData(const aly::Image4f& data) {
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] == ChannelType::data){
			getInput(i)->setValue(data);
			break;
		}
	}
}
void NeuralLayer::setInputData(const Tensor& data) {
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] == ChannelType::data){
			getInput(i)->value=data;
			break;
		}
	}
}
void NeuralLayer::setInputData(const std::vector<std::vector<const Storage*>>& data) {
	size_t n = 0;
	size_t cnt = data.size();
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] != ChannelType::data)continue;
		Tensor &dst_data = getInput(i)->value;
		assert(n < cnt);
		const std::vector<const Storage*>& storage = data[n++];
		size_t sz = storage.size();
		dst_data.resize(sz);
		for (size_t j = 0; j < sz; ++j) {
			dst_data[j] = *storage[j];
		}
	}
}
SignalPtr NeuralLayer::getInput(size_t i) {
	if (inputs[i].get() == nullptr) {
		inputs[i] = SignalPtr(
				new NeuralSignal(nullptr, getInputDimensions(i),
						inputTypes[i]));
	}
	return inputs[i];
}
SignalPtr NeuralLayer::getOutput(size_t i) {
	if (outputs[i].get() == nullptr) {
		outputs[i] = SignalPtr(
				new NeuralSignal(this, getOutputDimensions(i),outputTypes[i]));
	}
	return outputs[i];
}
std::vector<const Storage*> NeuralLayer::getInputWeights() const {
	std::vector<const Storage*> v;
	for (size_t i = 0; i < inputChannels; i++) {
		if (isTrainableWeight(inputTypes[i])) {
			v.push_back(getInput(i)->value.data());
		}
	}
	return v;
}
std::vector<const Storage*> NeuralLayer::getOutputWeights() const {
	std::vector<const Storage*> v;
	for (size_t i = 0; i < outputChannels; i++) {
		if (isTrainableWeight(outputTypes[i])) {
			v.push_back(getOutput(i)->value.data());
		}
	}
	return v;
}
std::vector<const Tensor*> NeuralLayer::getInputGradient() const {
	std::vector<const Tensor*> v;
	for (size_t i = 0; i < inputChannels; i++) {
		if (isTrainableWeight(inputTypes[i])) {
			v.push_back(&(getInput(i)->change));
		}
	}
	return v;
}
std::vector<const Tensor*> NeuralLayer::getOutputGradient() const {
	std::vector<const Tensor*> v;
	for (size_t i = 0; i < outputChannels; i++) {
		if (isTrainableWeight(outputTypes[i])) {
			v.push_back(&(getOutput(i)->change));
		}
	}
	return v;
}

std::vector< Storage*> NeuralLayer::getInputWeights()  {
	std::vector< Storage*> v;
	for (size_t i = 0; i < inputChannels; i++) {
		if (isTrainableWeight(inputTypes[i])) {
			v.push_back(getInput(i)->value.data());
		}
	}
	return v;
}
std::vector< Storage*> NeuralLayer::getOutputWeights()  {
	std::vector< Storage*> v;
	for (size_t i = 0; i < outputChannels; i++) {
		if (isTrainableWeight(outputTypes[i])) {
			v.push_back(getOutput(i)->value.data());
		}
	}
	return v;
}
std::vector< Tensor*> NeuralLayer::getInputGradient()  {
	std::vector< Tensor*> v;
	for (size_t i = 0; i < inputChannels; i++) {
		if (isTrainableWeight(inputTypes[i])) {
			v.push_back(&getInput(i)->change);
		}
	}
	return v;
}
std::vector< Tensor*> NeuralLayer::getOutputGradient()  {
	std::vector< Tensor*> v;
	for (size_t i = 0; i < outputChannels; i++) {
		if (isTrainableWeight(outputTypes[i])) {
			v.push_back(&getOutput(i)->change);
		}
	}
	return v;
}
void NeuralLayer::clearGradients() {
	for (int i = 0; i < inputChannels; i++) {
		getInput(i)->clearGradients();
	}
}
float NeuralLayer::getAspect() {
	aly::dim3 dims = getOutputSize();
	return (dims.x * dims.z * NeuralLayerRegion::GlyphSize
			+ (dims.z - 1) * NeuralLayerRegion::GlyphSpacing)
			/ (float) (dims.y * NeuralLayerRegion::GlyphSize);
}
size_t NeuralLayer::getInputDataSize() const {
 	size_t n = 0;
	for (size_t i = 0; i < inputChannels; i++) {
		if (inputTypes[i] != ChannelType::data)continue;
		n+= getInputDimensions(i).volume();
	}
	return n;
}
size_t NeuralLayer::getOutputDataSize() const {
 	size_t n = 0;
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputTypes[i] != ChannelType::data)continue;
		n+= getOutputDimensions(i).volume();
	}
	return n;
}

void NeuralLayer::updateWeights(
		NeuralOptimizer& optimizer,
		int batch_size) {
	float_t rcp_batch_size = float_t(1) / float_t(batch_size);
	auto &diff = weightDifference;
	for (int i = 0; i < inputChannels; i++) {
		if (trainable && isTrainableWeight(inputTypes[i])) {
			Storage& target = getInputWeights(i);
			getInput(i)->mergeGradients(diff);
			for (size_t j = 0; j < diff.size(); ++j) {
				diff[j] *= rcp_batch_size;
			}
			// parallelize only when target size is big enough to mitigate
			// thread spawning overhead.
			bool parallelize = (target.size() >= 512);
			optimizer.update(diff, target, parallelize);
		}
	}
	clearGradients();
	post();
}
aly::dim3 NeuralLayer::getInputSize() {
	if (inputSize.x < 0) {
		for (int i = 0; i < inputChannels; i++) {
			if (inputTypes[i] == ChannelType::data) {
				inputSize = getInputDimensions(i);
				break;
			}
		}
	}
	return inputSize;
}
aly::dim3 NeuralLayer::getOutputSize() {
	if (outputSize.x < 0) {
		for (int i = 0; i < outputChannels; i++) {
			if (outputTypes[i] == ChannelType::data) {
				outputSize = getOutputDimensions(i);
				break;
			}
		}
	}
	return outputSize;
}
bool NeuralLayer::hasSameWeights(const NeuralLayer &rhs, float_t eps) const {
	auto w1 = getInputWeights();
	auto w2 = rhs.getInputWeights();
	if (w1.size() != w2.size())
		return false;

	for (size_t i = 0; i < w1.size(); i++) {
		if (w1[i]->size() != w2[i]->size())
			return false;

		for (size_t j = 0; j < w1[i]->size(); j++) {
			if (std::abs(w1[i]->operator [](j) - w2[i]->operator [](j)) > eps)
				return false;
		}
	}
	return true;
}

void NeuralLayer::initializeWeights() {
	// layer/node is not trainable, do nothing and mark the layer/node
	// as initialized.
	if (!trainable) {
		initialized = true;
		return;
	}
	// Fill vector values with data generated by the initialization
	// function. The pointer to the data is obtained from the
	// computational graph and the methods fan_in_size() and fan_out_size()
	// return the number of incoming/outcoming connections for each
	// input/output unit.
	for (size_t i = 0; i < inputChannels; i++) {
		switch (inputTypes[i]) {
		// fill vectors of weight type
		case ChannelType::weight:
			if (weightInitFunc)
				weightInitFunc(getInputWeights(i), getFanInSize(),getFanOutSize());
			break;
			// fill vector of bias type
		case ChannelType::bias:
			if (biasInitFunc)
				biasInitFunc(getInputWeights(i), getFanInSize(),getFanOutSize());
			break;
		default:
			break;
		}
	}
	// in case we succeed with data initialization, we mark the
	// layer/node as initialized.
	initialized = true;
}
void NeuralLayer::setup(bool reset_weight) {
	// The input shape (width x height x depth) must be equal to the number
	// of input channels a.k.a the number of incoming vectors or 'edges' in
	// the computational nomenclature. Same is applied to output shape and
	// numbers of output edges.
	if (getInputDimensions().size() != inputChannels
			|| getOutputDimensions().size() != outputChannels) {
		throw std::runtime_error("Connection mismatch at setup layer");
	}
	// An 'edge' is created in the computational graph from the current
	// layer/node to each output node and allocates the needed memory.
	// The number of output nodes is determined by the layer interface.
	// In order to handle graph based networks, which a layer/node might
	// have multiple input/output connections, we need to check that the
	// connection edge does not already exists if we don't want duplicated
	// memory allocation.
	for (size_t i = 0; i < outputChannels; i++) {
		if (outputs[i].get() == nullptr) {
			std::cout<<"Setup "<<getName()<<" "<<getOutputDimensions(i)<<std::endl;
			outputs[i] = SignalPtr(new NeuralSignal(this, getOutputDimensions(i),outputTypes[i]));
		}
	}
	// reset the weights if necessary, or in case that the data is
	// still not initialized.
	if (reset_weight || !initialized) {
		initializeWeights();
	}
}

void NeuralLayer::expand() {
	std::shared_ptr<NeuralFlowPane> flowPane = sys->getFlow();
	box2px bounds = layerRegion->getBounds();
	int N = int(getOutputLayers().size());
	float layoutWidth = 0.0f;
	int C = 1;
	float width = 120.0f;
	const float MAX_WIDTH = 2048.0f;
	for (auto child : getOutputLayers()) {
		int c = child->getOutputSize().z;
		C = std::max(c, C);
		layoutWidth += (10.0f + std::min(width * c, MAX_WIDTH));
	}
	layoutWidth -= 10.0f;
	for (auto child : getOutputLayers()) {
		int c = child->getOutputSize().z;
		float offset = aly::round(0.5f * std::min(width * c, MAX_WIDTH));
		float height = child->getRegion()->setSize(
				std::min(width * c, MAX_WIDTH));
		float2 pos = pixel2(
				aly::round(bounds.position.x + bounds.dimensions.x * 0.5f- layoutWidth * 0.5f + offset),
				aly::round(bounds.position.y + bounds.dimensions.y + 0.5f * height + 10.0f));
		flowPane->add(child.get(), pos);
		offset += width * c + 10.0f;
	}
	flowPane->update();
}
bool NeuralLayer::isRoot() const {
	for (SignalPtr signal : inputs) {
		if (signal.get() != nullptr && signal->hasInput())
			return false;
	}
	return true;
}
bool NeuralLayer::isLeaf() const {
	for (SignalPtr signal : inputs) {
		if (signal.get() != nullptr && signal->hasOutput())
			return false;
	}
	return true;
}
void NeuralLayer::getNeuron(const aly::int3& pos, Neuron& neuron) {
	std::vector<int3> stencil;
	neuron.clear();
	for (int i = 0; i < inputChannels; i++) {
		if (inputTypes[i] == ChannelType::data) {
			SignalPtr data = getInput(i);
			if (data.get() != nullptr) {
				getStencilInput(pos, stencil);
				for (int3 st : stencil) {
					neuron.input.push_back(data->getValuePtr(st));
				}
			}
		} else if (inputTypes[i] == ChannelType::weight) {
			SignalPtr data = getInput(i);
			if (data.get() != nullptr) {
				getStencilWeight(pos, stencil);
				for (int3 st : stencil) {
					neuron.weights.push_back(data->getValuePtr(st));
				}
			}
		} else if (inputTypes[i] == ChannelType::bias) {
			SignalPtr data = getInput(i);
			if (data.get() != nullptr) {
				int3 st;
				if (getStencilBias(pos, st)) {
					neuron.bias = data->getValuePtr(st);
				}
			}
		}
	}
	for (int i = 0; i < outputChannels; i++) {
		if (outputTypes[i] == ChannelType::data) {
			SignalPtr data = getOutput(i);
			if (data.get() != nullptr) {
				neuron.output = data->getValuePtr(pos);
			}
		}
	}
}
void NeuralLayer::initialize(const aly::ExpandTreePtr& tree,
		const aly::TreeItemPtr& parent) {
	TreeItemPtr item;
	parent->addItem(item = TreeItemPtr(new TreeItem(getName(), 0x0f20e)));
	const float fontSize = 20;
	const int lines = getInputDimensionSize() + getOutputDimensionSize();
	item->addItem(
			LeafItemPtr(
					new LeafItem(
							[this,fontSize](AlloyContext* context, const box2px& bounds) {
								NVGcontext* nvg = context->nvgContext;
								float yoff = 2 + bounds.position.y;
								nvgFontSize(nvg, fontSize);
								nvgFontFaceId(nvg, context->getFontHandle(FontType::Normal));
								std::string label;
								std::vector<dim3> dims=getInputDimensions();
								for(int i=0;i<dims.size();i++) {
									label = MakeString() << "in."<<this->inputTypes[i]<<" "<<dims[i];
									drawText(nvg, bounds.position.x, yoff, label.c_str(), FontStyle::Normal, context->theme.LIGHTER);
									yoff += fontSize + 2;
								}
								dims=getOutputDimensions();
								for(int i=0;i<dims.size();i++) {
									label = MakeString() << "out."<<this->outputTypes[i]<<" "<<dims[i];
									drawText(nvg, bounds.position.x, yoff, label.c_str(), FontStyle::Normal, context->theme.LIGHTER);
									yoff += fontSize + 2;
								}
							}, pixel2(180, lines * (fontSize + 2) + 2))));
	item->onSelect = [this](TreeItem* item, const InputEvent& e) {
		sys->getFlow()->setSelected(this,e);
	};
	for (auto child : getOutputLayers()) {
		child->initialize(tree, item);
	}
}
}
