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
#ifndef _NeuralRuntime_H_
#define _NeuralRuntime_H_
#include <thread>
#include <mutex>
#include <chrono>
#include <AlloyParameterPane.h>
#include <AlloyWorker.h>
#include "NeuralSystem.h"
#include "NeuralCache.h"
#include "NeuralLossFunction.h"
#include "NeuralOptimizer.h"
namespace tgr {
class NeuralRuntime;
class NeuralListener {
public:
	virtual void NeuralEvent(NeuralRuntime* simulation, int simulationIteration,
			double time) = 0;
	virtual ~NeuralListener();
};
class NeuralRuntime: public aly::RecurrentTask {
protected:
	bool paused;
	bool isInitialized;
	double lastResidual;
	int threads;
	aly::Number iterationsPerEpoch;
	aly::Number iterationsPerStep;
	aly::Number learningRateInitial;
	aly::Number weightDecay;
	aly::Number momentum;
	aly::Number learningRateDelta;
	aly::Number batchSize;
	aly::Number minSample;
	aly::Number maxSample;
	aly::Number lowerSample;
	aly::Number upperSample;
	int optimizationMethod;
	int lossFunction;
	std::vector<int> sampleIndexes;
	std::vector<float> outputData;
	int iteration;
	std::thread simulationThread;
	std::shared_ptr<tgr::NeuralSystem> sys;
	std::shared_ptr<tgr::NeuralCache> cache;

	bool stop_training_;
	std::vector<Tensor> in_batch;
	std::vector<Tensor> t_batch;
	std::vector<Tensor> inputs;
	std::vector<Tensor> desiredOutputs;
	std::vector<Tensor> t_costs;
	NeuralOptimizer optimizer;
	NeuralLossFunction loss;
	const Tensor* get_target_cost_sample_pointer(
			const std::vector<Tensor> &t_cost, size_t i);
	void trainOnce(NeuralOptimizer &optimizer, const NeuralLossFunction& loss,const Tensor *in, const Tensor *t, int size, const int nbThreads,const Tensor *t_cost);
	void trainOneBatch(NeuralOptimizer &optimizer,const NeuralLossFunction& loss, const Tensor *in, const Tensor *t,int batch_size, const int num_tasks, const Tensor *t_cost);
public:
	float getLoss(const NeuralLossFunction& loss);
	std::function<void(int iteration, bool lastIteration)> onUpdate;
	std::function<void()> onBatchEnumerate;
	std::function<void()> onEpochEnumerate;
	typedef std::chrono::high_resolution_clock Clock;
	bool step();
	bool init(bool reset_weights = false);
	void setBatchSize(int b) {
		batchSize.setValue(b);
	}
	void setData(const std::vector<Tensor>& input,
			std::vector<Tensor>& desiredOutputs,
			const std::vector<Tensor> &t_cost = std::vector<Tensor>());
	void setData(const std::vector<Storage> &inputs,
			const std::vector<int> &class_labels,
			const std::vector<Storage>& t_cost = std::vector<Storage>());
	void setData(const std::vector<Tensor> &inputs,
			const std::vector<int> &class_labels,
			const std::vector<Storage>& t_cost = std::vector<Storage>());
	void setLossFunction(const NeuralLossFunction& loss) {
		this->loss = loss;
	}
	void setOptimizer(const NeuralOptimizer& opt) {
		this->optimizer = opt;
	}
	void cleanup();
	std::shared_ptr<tgr::NeuralCache> getCache() const {
		return cache;
	}
	void setSampleRange(int mn, int mx);
	void setSelectedSamples(int mn, int mx);
	void setup(const aly::ParameterPanePtr& pane);
	NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system);
	uint64_t getMaxIteration() const {
		return uint64_t(iterationsPerEpoch.toInteger());
	}
	int getIterationsPerEpoch() const {
		return iterationsPerEpoch.toInteger();
	}
	int getIterationsPerStep() const {
		return iterationsPerStep.toInteger();
	}
	uint64_t getIteration() const {
		return iteration;
	}
	virtual ~NeuralRuntime() {
	}
	;
};
typedef std::shared_ptr<NeuralRuntime> NeuralRuntimePtr;
}
#endif
