#ifndef _NEURAL_SYSTEM_H_
#define _NEURAL_SYSTEM_H_
#include "NeuralLayer.h"
#include <map>
namespace tgr {
	class NeuralFilter;
	class NeuralSystem {
	protected:
		std::set<SignalPtr> signals;
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

		const std::vector<float>& getOutput() const {
			return output;
		}
		const std::vector<float>& getInput() const {
			return input;
		}
		void setInput(const std::vector<float>& in) {
			input = in;
		}
		void add(const SignalPtr& signal);
		SignalPtr add(Terminal source,Terminal target,float weight=0.0f);
		SignalPtr connect(int si,int sj,const NeuralLayerPtr& sl, int ti, int tj, const NeuralLayerPtr& tl, float weight = 0.0f);
		SignalPtr connect(int si, int sj,int sb, const NeuralLayerPtr& sl, int ti, int tj,int tb, const NeuralLayerPtr& tl, float weight = 0.0f);
		void add(const std::shared_ptr<NeuralFilter>& filter);
	
	};
}
#endif