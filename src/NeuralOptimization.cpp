

#include "NeuralOptimization.h"
namespace tgr {
	bool GradientDescentOptimizer::optimize(const std::vector<std::shared_ptr<Signal>>& signals) {
		int N = (int)signals.size();
		double delta = 0.0f;
		for (int n = 0; n < N;n++) {
			SignalPtr sig = signals[n];
			delta += std::abs(sig->change);
			sig->value -= sig->change*learningRate;
		}
		std::cout << "Optimize Signals " << signals.size() <<" "<<delta<< std::endl;
		return true;
	}
}