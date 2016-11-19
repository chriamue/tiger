#ifndef _NEURAL_SYSTEM_H_
#define _NEURAL_SYSTEM_H_
#include "NeuronLayer.h"
#include <map>
namespace tgr {
	class NeuralSystem {
	protected:
		std::vector<NeuronLayerPtr> layers;
		std::vector<SignalPtr> signals;
		std::vector<float> inputs;
		std::vector<Terminal> inputTerminals;
		std::vector<float> outputs;
		std::vector<Terminal> outputTerminals;
		std::vector<std::vector<SignalPtr>> forwardNetwork;
	public:
		void evaluate();
		void initialize();
		void train(float learningRate);
		const std::vector<float>& getOutput() const {
			return outputs;
		}
		const std::vector<float>& getInput() const {
			return inputs;
		}
		void setInput(const std::vector<float>& in) {
			inputs = in;
		}
		void add(const SignalPtr& signal);
		SignalPtr add(Terminal source,Terminal target,float weight=0.0f);
		void add(const NeuronLayerPtr& layer);
	};
}
#endif