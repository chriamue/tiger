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
	void WriteNeuralStateToFile(const std::string& file, const NeuralState& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ofstream os(file);
			cereal::JSONOutputArchive archive(os);
			archive(cereal::make_nvp("neuralstate", params));
		}
		else if (ext == "xml") {
			std::ofstream os(file);
			cereal::XMLOutputArchive archive(os);
			archive(cereal::make_nvp("neuralstate", params));
		}
		else {
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
		}
		else if (ext == "xml") {
			std::ifstream os(file);
			cereal::XMLInputArchive archive(os);
			archive(cereal::make_nvp("neuralstate", params));
		}
		else {
			std::ifstream os(file, std::ios::binary);
			cereal::PortableBinaryInputArchive archive(os);
			archive(cereal::make_nvp("neuralstate", params));
		}
	}
	void NeuralLayer::set(const Knowledge& k, const Knowledge& bk) {
		weights = k;
		biasWeights = bk;
	}
	void NeuralLayer::setState(const NeuralState& state) {
		name = state.name;
		weights.set(state.weights);
		weightChanges.set(state.weightChanges);
		biasWeights.set(state.biasWeights);
		biasWeightChanges.set(state.biasWeightChanges);
		responses.set(state.responses);
		responseChanges.set(state.responseChanges);
		biasResponses.set(state.biasResponses);
		biasResponseChanges.set(state.biasResponseChanges);
	}
	NeuralState NeuralLayer::getState() const {
		NeuralState state;
		state.name = name;
		state.weights = weights;
		state.weightChanges = weightChanges;
		state.biasWeights = biasWeights;
		state.biasWeightChanges = biasWeightChanges;
		state.responses = responses;
		state.responseChanges = responseChanges;
		state.biasResponses = biasResponses;
		state.biasResponseChanges = biasResponseChanges;
		return state;
	}
	void NeuralLayer::setId(int i) {
		id = i;
		/*
		for (Neuron& n : neurons) {
			n.id.z = id;
		}
		for (Neuron& n : biasNeurons) {
			n.id.z = id;
		}
		*/
	}
	NeuralLayer::NeuralLayer(int width, int height, int bins, bool bias, const NeuronFunction& func) :
			bias(bias),compiled(false),visited(false),trainable(true),residualError(0.0),id(-1),width(width), height(height), depth(bins){
		neurons.resize(width*height*bins, Neuron(func));
		graph.reset(new GraphData(getName()));
	}
	std::shared_ptr<aly::NeuralFlowPane> NeuralLayer::getFlow() const {
		return sys->getFlow();
	}
	void NeuralLayer::initializeWeights(float minW, float maxW) {
		for (SignalPtr sig : signals) {
			*sig->weight=RandomUniform(minW, maxW);
		}
	}
	void NeuralLayer::reset() {
		residualError = 0.0;
		for (Neuron& neuron : neurons) {
			*neuron.change = 0.0f;
			*neuron.value = 0.0f;
		}
		for (Neuron& neuron : biasNeurons) {
			*neuron.change = 0.0f;
		}
		for (SignalPtr sig : signals) {
			*sig->change=0.0f;
		}
	}
	NeuralLayer::NeuralLayer(const std::string& name,int width, int height, int bins,bool bias, const NeuronFunction& func) :
			name(name), bias(bias), compiled(false),visited(false), trainable(true), residualError(0.0),id(-1), width(width), height(height), depth(bins) {
		neurons.resize(width*height*bins,Neuron(func));
		graph.reset(new GraphData(getName()));
	}
	double NeuralLayer::accumulate(double r) {
		residualError += r;
		
		return residualError;
	}
	void NeuralLayer::resize(int w, int h, int b) {
		neurons.resize(w * h * b);
		neurons.shrink_to_fit();
		width = w;
		height = h;
		depth = b;
	}
	void NeuralLayer::backpropagate() {
		int N = (int)neurons.size();
		//double residual = 0.0;
		//reduction(+:residual)
#pragma omp parallel for
		for (int n = 0; n < N; n++) {
			neurons[n].backpropagate();
		}
        //		residual /= N;
		//std::cout << "Backprop [" << getName() << "|" << N << "] Residual="<<residual << std::endl;
	}
	void NeuralLayer::setRegionDirty(bool b) {
		if (layerRegion.get() != nullptr) {
			layerRegion->setDirty(b);
		}
	}
	void NeuralLayer::evaluate() {
		int N = (int)neurons.size();
		double mag = 0.0;
#pragma omp parallel for reduction(+:mag)
		for (int n = 0; n < N; n++) {
			mag+=std::abs(neurons[n].evaluate());
		}
		mag /= N;
		//std::cout << "Evaluate [" << getName() << "|" << N << "] Magnitude=" << mag << std::endl;
		if (layerRegion.get() != nullptr) {
			layerRegion->setDirty(true);
		}
	}
	aly::Vector1f NeuralLayer::toVector() const{
		int N = (int)neurons.size();
		Vector1f data(N);
		for (int n = 0; n < N; n++) {
			data[n] = neurons[n].value;
		}
		return data;
	}
	struct SignalCompare {
		bool operator()(const SignalPtr& lhs, const SignalPtr& rhs) const
		{
			return lhs->id< rhs->id;
		}
	};
	void NeuralLayer::compile() {
		if (compiled)return;
		signals.clear();
		std::set<SignalPtr, SignalCompare> tmp;
		for (Neuron& n : neurons) {
			for (SignalPtr sig : n.getInput()) {
				tmp.insert(sig);
			}
		}
		signals.insert(signals.begin(), tmp.begin(), tmp.end());
		size_t N = tmp.size();
		size_t n = 0;
		weights.resize(N);
		weightChanges.resize(N);
		for (SignalPtr sig:tmp) {
			sig->weight = &weights[n];
			sig->change = &weightChanges[n];
			n++;
		}
		tmp.clear();
		N = neurons.size();
		responses.resize(N);
		responseChanges.resize(N);
		for (size_t n = 0; n < N; n++) {
			Neuron& neuron = neurons[n];
			neuron.value = &responses[n];
			neuron.change = &responseChanges[n];
			*neuron.value = 0.0f;
			/*
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					Neuron& n = neurons[i + j*width];
					n.id.x = i;
					n.id.y = j;
				}
			}
			*/
		}
		
		if (bias) {
			size_t N = width*height;
			biasNeurons.resize(N, Bias());
			biasWeights.resize(N);
			biasWeightChanges.resize(N);
			biasResponses.resize(N);
			biasResponseChanges.resize(N);

			/*
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					Neuron& n = biasNeurons[i+j*width];
					n.id.x = i;
					n.id.y = j;
				}
			}
			*/
			for (size_t n = 0; n < N; n++) {
				Neuron& neuron = biasNeurons[n];
				SignalPtr sig = MakeConnection(&neuron, &neurons[n]);
				sig->weight = &biasWeights[n];
				sig->change = &biasWeightChanges[n];
				neuron.value = &biasResponses[n];
				neuron.change = &biasResponseChanges[n];
				*neuron.value = 1.0f;
				signals.push_back(sig);
			}
		}
		compiled = true;
	}
	bool NeuralLayer::optimize() {
		if (optimizer.get() != nullptr) {
			return optimizer->optimize(id,signals);
		}
		else {
			//std::cerr << "No optimizer for " << getName() << std::endl;
			return false;
		}
	}
	void NeuralLayer::addChild(const std::shared_ptr<NeuralLayer>& layer) {
		children.push_back(layer);
		layer->dependencies.push_back(this);
	}
	void NeuralLayer::setFunction(const NeuronFunction& func) {
		for (Neuron& n : neurons) {
			n.setFunction(func);
		}
	}
	int NeuralLayer::getBin(size_t index) const {
		return clamp((int)std::floor(*neurons[index].value*depth), 0, depth-1);
	}
	int NeuralLayer::getBin(const Neuron& n) const {
		return clamp((int)std::floor(*n.value*depth), 0, depth-1);
	}
	
	const Neuron& NeuralLayer::operator[](const size_t i) const {
		return neurons[i];
	}
	Neuron& NeuralLayer::operator[](const size_t i) {
		return neurons[i];
	}
	const Neuron* NeuralLayer::get(const int i, const int j) const {
		if(neurons.size()==0)throw std::runtime_error("Neurons not initialized.");
		return &neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	Neuron* NeuralLayer::get(const int i, const int j) {
		if(neurons.size()==0)throw std::runtime_error("Neurons not initialized.");
		return &neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	const Neuron* NeuralLayer::get(const size_t i) const {
		if(neurons.size()==0)throw std::runtime_error("Neurons not initialized.");
		return &neurons[i];
	}
	Neuron* NeuralLayer::get(const size_t i) {
		if(neurons.size()==0)throw std::runtime_error("Neurons not initialized.");
		return &neurons[i];
	}

	Neuron& NeuralLayer::operator()(const int i, const int j) {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const size_t i, const size_t j) {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const aly::int2 ij) {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const Terminal ij) {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const int i, const int j) const {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const size_t i, const size_t j) const {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const aly::int2 ij) const {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const Terminal ij) const {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	bool NeuralLayer::visitedChildren() const {
		for (NeuralLayerPtr layer : children) {
			if (!layer->isVisited())return false;
		}
		return true;
	}
	bool NeuralLayer::visitedDependencies() const {
		for (NeuralLayer* layer : dependencies) {
			if (!layer->isVisited())return false;
		}
		return true;
	}
	bool NeuralLayer::isVisible() const {
		if (layerRegion.get() != nullptr&&layerRegion->parent!=nullptr) {
			return layerRegion->isVisible();
		}
		else {
			return false;
		}
	}
	aly::NeuralLayerRegionPtr NeuralLayer::getRegion() {
		if (layerRegion.get() == nullptr) {
			
			float2 dims=float2(240.0f,240.0f/ getAspect())+ NeuralLayerRegion::getPadding();
			layerRegion = NeuralLayerRegionPtr(new NeuralLayerRegion(name,this, CoordPerPX(0.5f, 0.5f, -dims.x*0.5f, -dims.y*0.5f), CoordPX(dims.x, dims.y)));
			if (hasChildren()) {
				layerRegion->setExpandable(true);
				for (auto child : getChildren()) {
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
	void NeuralLayer::expand() {
		std::shared_ptr<NeuralFlowPane> flowPane = sys->getFlow();
		box2px bounds = layerRegion->getBounds();
		int N = int(getChildren().size());
		float layoutWidth = 0.0f;
		float width = 120.0f;
		float offset = 0.5f*width;
		layoutWidth = (10.0f + width)*N - 10.0f;
		for (auto child : getChildren()) {
			float height = child->getRegion()->setSize(width);
			float2 pos = pixel2(
				bounds.position.x + bounds.dimensions.x*0.5f - layoutWidth*0.5f + offset,
				bounds.position.y + bounds.dimensions.y + 0.5f*height + 10.0f);
			flowPane->add(child.get(), pos);
			offset += width + 10.0f;
		}
		flowPane->update();
	}
	void NeuralLayer::set(const Image1f& input) {
		if(input.width!=width||input.height!=height){
			throw std::runtime_error(MakeString()<<"Image dimensions ("<<input.width<<","<<input.height<<") do not match neuron dimensions ("<<width<<","<<height<<")");
		}
		for (int j = 0; j < input.height; j++) {
			for (int i = 0; i < input.width; i++) {
				float* ptr=get(i, j)->value;
				*ptr = input(i, j).x;
			}
		}
		if (layerRegion.get() != nullptr) {
			layerRegion->setDirty(true);
		}
	}
	void NeuralLayer::set(const std::vector<float>& input) {
		if(input.size()!=neurons.size()){
			throw std::runtime_error(MakeString()<<"Vector dimension ("<<input.size()<<") do not match neuron dimension ("<<neurons.size()<<")");
		}
		for (size_t i = 0; i < input.size(); i++) {
			*get(i)->value = input[i];
		}
		if (layerRegion.get() != nullptr) {
			layerRegion->setDirty(true);
		}
	}
	void NeuralLayer::get( Image1f& input) {
		input.resize(width, height);
		for (int j = 0; j < input.height; j++) {
			for (int i = 0; i < input.width; i++) {
				input(i, j).x = *get(i, j)->value;
			}
		}
	}
	void NeuralLayer::get(std::vector<float>& input) {
		input.resize(size());
		for (size_t i = 0; i < input.size(); i++) {
			input[i] = *get(i)->value;
		}
	}

	void NeuralLayer::initialize(const aly::ExpandTreePtr& tree, const aly::TreeItemPtr& parent)  {
		TreeItemPtr item;
		parent->addItem(item=TreeItemPtr(new TreeItem(getName(), 0x0f20e)));
		const float fontSize = 20;
		const int lines = 2;
		item->addItem(LeafItemPtr(new LeafItem([this,fontSize](AlloyContext* context, const box2px& bounds) {
			NVGcontext* nvg = context->nvgContext;
			float yoff = 2 + bounds.position.y;
			nvgFontSize(nvg, fontSize);
			nvgFontFaceId(nvg, context->getFontHandle(FontType::Normal));
			std::string label;

			label = MakeString() << "In Layers: " << getDependencies().size() <<" Out Layers: "<<getChildren().size();
			drawText(nvg, bounds.position.x, yoff, label.c_str(), FontStyle::Normal, context->theme.LIGHTER);
			yoff += fontSize + 2;

			label = MakeString() << "Size: " << width << " x " << height << " x " << depth;
			drawText(nvg, bounds.position.x, yoff, label.c_str(), FontStyle::Normal, context->theme.LIGHTER);
			yoff += fontSize + 2;

		}, pixel2(180, lines*(fontSize + 2) + 2))));
		item->onSelect = [this](TreeItem* item, const InputEvent& e) {
			sys->getFlow()->setSelected(this,e);

		};
		for (auto child : children) {
			child->initialize(tree, item);
		}
	}

}
