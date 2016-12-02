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
		std::cout << "Intialize " << std::endl;
		sys->setOptimizer(std::shared_ptr<NeuralOptimization>(new GradientDescentOptimizer(0.8f)));
		iteration = 0;
		return true;
	}
	void NeuralRuntime::cleanup(){
	}
	void NeuralRuntime::setup(const aly::ParameterPanePtr& controls) {
		controls->addGroup("Training", true);
		controls->addSelectionField("Optimizer", optimizationMethod, std::vector<std::string>{"Gradient Descent", "Momentum"},6.0f);
		controls->addNumberField("Epochs", epochs);
		controls->addNumberField("Outer Iterations", iterationsPerEpoch);
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
				//std::cout << "Index " << idx << ": " << err << std::endl;
			}
			sys->backpropagate();
		}
		
		res /= (sampleIndexes.size());
		std::cout << "Residual " <<res << std::endl;
		sys->optimize();
		if (iter%uint64_t(iterationsPerStep.toInteger())==0&&onUpdate) {
			onUpdate(iter, !ret);
		}
		iteration++;
		if (iteration >= getMaxIteration()) {
			ret = false;
		}
		return ret;
	}
	NeuralRuntime::NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system) :
		RecurrentTask([this](uint64_t iteration) {return step();}, 5),sys(system),paused(false) {
		optimizationMethod = 0;
		epochs = Integer(12);
		iterationsPerEpoch = Integer(100);
		iterationsPerStep = Integer(10);
		learningRateInitial = Float(0.8f);
		learningRateDelta = Float(0.5);
	}
}