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
		std::vector<NeuralLayerPtr> roots;
		std::map<Terminal,float> input;
		std::map<Terminal,float> output;
	public:
		void evaluate();
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