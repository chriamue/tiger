#include "NeuralOptimization.h"
namespace tgr {
	bool GradientDescentOptimizer::optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) {
		int N = (int)signals.size();
#pragma omp parallel for
		for (int n = 0; n < N;n++) {
			SignalPtr sig = signals[n];
			float* w=sig->weight;
			*w -= learningRate*(*sig->change +weightDecay*(*w));
		}
		//std::cout << "Weight change="<<delta<< std::endl;
		return true;
	}	
	bool MomentumOptimizer::optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) {
		int N = (int)signals.size();
		auto pos = velocityBufferMap.find(id);
		if (pos == velocityBufferMap.end()) {
			velocityBufferMap[id]=std::vector<float>(signals.size(), 0.0f);
		}
		std::vector<float>& velocityBuffer = velocityBufferMap.at(id);
#pragma omp parallel for
		for (int n = 0; n < N; n++) {
			SignalPtr sig = signals[n];
			float prev = velocityBuffer[n];
			float* w=sig->weight;
			float vel = momentum * prev - learningRate* (*sig->change + (*w) * weightDecay);
			*w += vel;
			velocityBuffer[n] = vel;
		}
		return true;
	}
}