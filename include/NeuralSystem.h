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
		std::vector<SignalPtr> signals;
		std::vector<float> input;
		std::vector<Terminal> inputTerminals;
		std::vector<float> output;
		std::vector<Terminal> outputTerminals;
		std::vector<std::vector<SignalPtr>> forwardNetwork;
	public:
		void evaluate();
		void initialize();
		void train(float learningRate);
		Neuron* getNeuron(const Terminal& t) const;
		void initialize(const aly::ExpandTreePtr& tree);
		const std::vector<float>& getOutput() const {
			return output;
		}
		const std::vector<float>& getInput() const {
			return input;
		}
		void setInput(const std::vector<float>& in) {
			input = in;
		}
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
		void add(const SignalPtr& signal);
		void add(const std::vector<SignalPtr>& signals);
		SignalPtr add(Terminal source,Terminal target,float weight=0.0f);
		SignalPtr connect(int si,int sj,const NeuralLayerPtr& sl, int ti, int tj, const NeuralLayerPtr& tl, float weight = 0.0f);
		SignalPtr connect(int si, int sj,int sb, const NeuralLayerPtr& sl, int ti, int tj,int tb, const NeuralLayerPtr& tl, float weight = 0.0f);
		void add(const std::shared_ptr<NeuralFilter>& filter);
	
	};
}
#endif