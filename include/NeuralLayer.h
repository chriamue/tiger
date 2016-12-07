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
#include "Neuron.h"
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
	std::string MakeID(int len=8);
	class NeuralSystem;
	class NeuralLayer {
		protected:
			std::vector<Neuron> neurons;
			std::vector<Neuron> biasNeurons;
			std::vector<std::shared_ptr<Signal>> signals;

			std::vector<std::shared_ptr<NeuralLayer>> children;
			std::vector<NeuralLayer*> dependencies;
			std::shared_ptr<NeuralOptimization> optimizer;
			std::string name;
			
			bool bias;
			bool visited;
			bool trainable;
			double residualError;
			aly::NeuralLayerRegionPtr layerRegion;
			aly::GraphDataPtr graph;
			
			NeuralSystem* sys;
			int id;
		public:
			int width;
			int height;
			int bins;
			Knowledge weights;
			Knowledge weightChanges;
			Knowledge biasWeights;
			Knowledge biasWeightChanges;
			Knowledge responses;
			Knowledge responseChanges;
			Knowledge biasResponses;
			Knowledge biasResponseChanges;
			typedef Neuron ValueType;
			typedef typename std::vector<ValueType>::iterator iterator;
			typedef typename std::vector<ValueType>::const_iterator const_iterator;
			typedef typename std::vector<ValueType>::reverse_iterator reverse_iterator;
			int getId() const {
				return id;
			}
			void setId(int i) {
				id = i;
			}
			void set(const Knowledge& k);
			Knowledge& getWeights() {
				return weights;
			}
			Knowledge& getResponses() {
				return responses;
			}
			Knowledge& getWeightChanges() {
				return weightChanges;
			}
			Knowledge& getResponseChanges() {
				return responseChanges;
			}
			const Knowledge& getWeights() const {
				return weights;
			}
			const Knowledge& getResponses() const {
				return responses;
			}
			const Knowledge& getWeightChanges() const {
				return weightChanges;
			}
			const Knowledge& getResponseChanges() const {
				return responseChanges;
			}
			std::shared_ptr<aly::NeuralFlowPane> getFlow() const;
			iterator begin() {
				return neurons.begin();
			}
			iterator end() {
				return neurons.end();
			}
			aly::GraphDataPtr getGraph() const {
				return graph;
			}
			void expand();
			void setResidual(float r) {
				residualError = r;
			}
			double getResidual() const {
				return residualError;
			}
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
			bool optimize();
			void update();
			void compile();
			void evaluate();
			void reset();
			void initializeWeights(float minW=0.0f, float maxW=1.0f);

			void backpropagate();
			aly::NeuralLayerRegionPtr getRegion();
			bool hasRegion() const {
				return (layerRegion.get() != nullptr&&layerRegion->parent!=nullptr);
			}
			bool isVisible() const; 
			std::vector<std::shared_ptr<Signal>>& getSignals() {
				return signals;
			}
			const std::vector<std::shared_ptr<Signal>>& getSignals() const {
				return signals;
			}
			std::vector<SignalPtr> getBiasSignals() const;
			std::vector<std::shared_ptr<NeuralLayer>>& getChildren() {
				return children;
			}
			std::vector<NeuralLayer*>& getDependencies() {
				return dependencies;
			}
			bool visitedChildren() const;
			bool visitedDependencies() const;

			const std::vector<std::shared_ptr<NeuralLayer>>& getChildren() const {
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
			std::string getName()const {
				return name;
			}

			void initialize(const aly::ExpandTreePtr& tree,const aly::TreeItemPtr& treeItem);
			void setFunction(const NeuronFunction& func);
			int getBin(size_t index) const;
			int getBin(const Neuron& n) const;

			aly::int2 dimensions() const {
				return aly::int2(width, height);
			}
			float getAspect() const {
				return width / (float)height;
			}
			const Neuron* get(int i, int j) const;
			Neuron* get(int i, int j);
			void resize(int width, int height, int b=1);
			const Neuron& operator[](const size_t i) const;
			size_t size() const {
				return neurons.size();
			}
			Neuron& operator[](const size_t i);
			Neuron& operator()(const int i, const int j);
			Neuron& operator()(const size_t i, const size_t j);
			Neuron& operator()(const aly::int2 ij);
			Neuron& operator()(const Terminal ij);
			const Neuron* get(const size_t i) const;
			Neuron* get(const size_t i);
			const Neuron& operator()(const int i, const int j) const;
			const Neuron& operator()(const size_t i, const size_t j) const;
			const Neuron& operator()(const aly::int2 ij) const;
			const Neuron& operator()(const Terminal ij) const;
			void set(const aly::Image1f& input);
			void set(const std::vector<float>& input);
			void get(aly::Image1f& input);
			void get(std::vector<float>& input);
			aly::Vector1f toVector() const;
			NeuralLayer() {}
			NeuralLayer(int width,int height,int bins,bool bias=false, const NeuronFunction& func = ReLU());
			NeuralLayer(const std::string& name,int width, int height, int bins, bool bias = false, const NeuronFunction& func=ReLU());
	};
	typedef std::shared_ptr<NeuralLayer> NeuralLayerPtr;
}
#endif