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
#ifndef _NEURALOPTIMIZATION_H_
#define _NEURALOPTIMIZATION_H_
#include <Signal.h>
namespace tgr {
	enum class NeuralOptimizer{GradientDescent,GradientMomentum};
	struct NeuralOptimization {
	protected:
		float learningRate;
	public:
		float getLearningRate() const {
			return learningRate;
		}
		void setLearningRate(float rate) {
			learningRate = rate;
		}
		NeuralOptimization(float learningRate) :learningRate(learningRate) {
		}
		virtual NeuralOptimizer getType() const = 0;
		virtual bool optimize(int id,const std::vector<std::shared_ptr<Signal>>& signals) = 0;
	};
	typedef std::shared_ptr<NeuralOptimization> NeuralOptimizationPtr;
	class GradientDescentOptimizer:public NeuralOptimization {
	protected:
		float weightDecay;
	public:
		GradientDescentOptimizer(float learningRate,float weightDecay=0.0f) :NeuralOptimization(learningRate),weightDecay(weightDecay) {
		}
		NeuralOptimizer getType() const {
			return NeuralOptimizer::GradientDescent;
		}
		virtual bool optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) override;
	};

	class MomentumOptimizer :public NeuralOptimization {
	protected:
		float weightDecay;
		float momentum;
		std::map<int,std::vector<float>> velocityBufferMap;
	public:
		MomentumOptimizer(float learningRate, float weightDecay=0.0f,float momentum =0.9f) :NeuralOptimization(learningRate), weightDecay(weightDecay), momentum(momentum) {
		}
		NeuralOptimizer getType() const {
			return NeuralOptimizer::GradientDescent;
		}
		virtual bool optimize(int id, const std::vector<std::shared_ptr<Signal>>& signals) override;
	};
}
#endif