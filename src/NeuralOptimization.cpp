#include "NeuralOptimization.h"
namespace tgr {
	bool GradientDescentOptimizer::optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) {
		int N = (int)signals.size();
		double delta = 0.0;
#pragma omp parallel for reduction(+:residual)
		for (int n = 0; n < N;n++) {
			SignalPtr sig = signals[n];
			float* w=sig->weight;
			delta += std::abs(*sig->change);
			*w = *w - learningRate*(*sig->change + weightDecay*(*w));
		}
		//if (N>0)std::cout <<"["<<id<<"] Weight Change="<<delta<< std::endl;
		return true;
	}	
	bool MomentumOptimizer::optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) {
		int N = (int)signals.size();
		auto pos = velocityBufferMap.find(id);
		if (pos == velocityBufferMap.end()) {
			velocityBufferMap[id]=std::vector<float>(signals.size(), 0.0f);
		}
		double delta = 0.0;
		std::vector<float>& velocityBuffer = velocityBufferMap.at(id);
#pragma omp parallel for reduction(+:residual)
		for (int n = 0; n < N; n++) {
			SignalPtr sig = signals[n];
			float prev = velocityBuffer[n];
			float* w=sig->weight;
			float vel = momentum * prev - learningRate* (*sig->change + (*w) * weightDecay);
			*w = *w + vel;
			delta += std::abs(*sig->change);
			velocityBuffer[n] = vel;
		}
		delta /= N;
		//if(N>0)std::cout << "[" << id << "] Weight Change=" << delta <<" signals "<<signals.size()<< std::endl;
		return true;
	}
}