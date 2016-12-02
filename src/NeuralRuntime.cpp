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
#include "NeuralRuntime.h"
#include <sstream>
#include <fstream>
#include <ostream>
using namespace aly;
namespace tgr {
	NeuralListener::~NeuralListener() {

	}
	bool NeuralRuntime::init() {
		lastResidual = 1E30f;
		for (NeuralLayerPtr layer : sys->getLayers()) {
			layer->getGraph()->points.clear();
		}
		sys->setOptimizer(opt=std::shared_ptr<NeuralOptimization>(new GradientDescentOptimizer(learningRateInitial.toFloat())));
		iteration = 0;
		return true;
	}
	void NeuralRuntime::cleanup(){
	}
	void NeuralRuntime::setup(const aly::ParameterPanePtr& controls) {
		controls->addGroup("Training", true);
		controls->addSelectionField("Optimizer", optimizationMethod, std::vector<std::string>{"Gradient Descent", "Momentum"},6.0f);
		controls->addNumberField("Epochs", iterationsPerEpoch);
		controls->addNumberField("Inner Iterations", iterationsPerStep);
		controls->addNumberField("Learning Rate", learningRateInitial, Float(0.0f), Float(1.0f));
		controls->addNumberField("Learning Delta", learningRateDelta, Float(0.0f), Float(1.0f));
		
	}
	bool NeuralRuntime::step() {
		uint64_t iter =iteration;
		bool ret = true;
		double res = 0;
		sys->reset();
		for (int idx : sampleIndexes) {
			if (inputSampler)inputSampler(sys->getInput(), idx);
			sys->evaluate();
			if (outputSampler) {
				outputSampler(outputData, idx);
				double err= sys->accumulate(outputData);
				res += err;
			}
			sys->backpropagate();
		}
		res /= (sampleIndexes.size());
		double delta = std::abs(lastResidual - res);
		if (delta < 1E-3f) {
			opt->setLearningRate(opt->getLearningRate()*learningRateDelta.toFloat());
			
		}
		if (iter%iterationsPerStep.toInteger() == 0) {
			std::cout << "Error=" << res << " Learning Rate=" << opt->getLearningRate() << std::endl;
		}
		lastResidual = res;
		sys->optimize();
		for (NeuralLayerPtr layer : sys->getLayers()) {
			layer->getGraph()->points.push_back(float2(float(iter),float(layer->getResidual())));
		}
		if (delta<1E-7f) {
			ret = false;
		}
		if (onUpdate) {
			onUpdate(iter, !ret);
		}
		iteration++;

		return ret;
	}
	NeuralRuntime::NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system) :
		RecurrentTask([this](uint64_t iteration) {return step();}, 5),sys(system),paused(false) {
		optimizationMethod = 0;
		iterationsPerEpoch = Integer(20);
		iterationsPerStep = Integer(10);
		learningRateInitial = Float(0.999f);
		learningRateDelta = Float(0.9f);
	}
}