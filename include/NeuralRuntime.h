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
namespace tgr {
	class NeuralRuntime;
	class NeuralListener {
	public:
		virtual void NeuralEvent(NeuralRuntime* simulation, int mSimulationIteration, double time) = 0;
		virtual ~NeuralListener();
	};
	class NeuralRuntime : public aly::RecurrentTask {
	protected:
		bool paused;
		bool isInitialized;
		double lastResidual;
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
		std::shared_ptr<NeuralOptimization> opt;
		int optimizationMethod;

		std::vector<int> sampleIndexes;
		std::vector<float> outputData;
		int iteration;
		std::thread simulationThread;
		std::shared_ptr<tgr::NeuralSystem> sys;

		std::shared_ptr<tgr::NeuralCache> cache;
	public:
		std::function<void(int iteration, bool lastIteration)> onUpdate;
		std::function<void(const NeuralLayerPtr& input,int idx)> inputSampler;
		std::function<void(std::vector<float>& outputData, int idx)> outputSampler;
		typedef std::chrono::high_resolution_clock Clock;
		bool step();
		bool init();
		void cleanup();
		std::shared_ptr<tgr::NeuralCache> getCache() const {
			return cache;
		}
		void setSamples(int mn, int mx);
		void setup(const aly::ParameterPanePtr& pane);
		NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system);
		uint64_t getMaxIteration() const {
			return uint64_t(iterationsPerEpoch.toInteger())*iterationsPerStep.toInteger();
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
		virtual ~NeuralRuntime() {};
	};
	typedef std::shared_ptr<NeuralRuntime> NeuralRuntimePtr;
}
#endif