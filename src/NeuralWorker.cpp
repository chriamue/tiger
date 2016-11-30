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
#include "NeuralWorker.h"
#include <sstream>
#include <fstream>
#include <ostream>
using namespace aly;
namespace tgr {
	NeuralListener::~NeuralListener() {

	}
	bool NeuralWorker::init() {
		sys->setOptimizer(std::shared_ptr<NeuralOptimization>(new GradientDescentOptimizer(0.5f)));
		iteration = 0;
		return false;
	}
	void NeuralWorker::cleanup(){
	}
	void NeuralWorker::setup(const aly::ParameterPanePtr& controls) {
		controls->addGroup("Training", true);
		controls->addSelectionField("Optimizer", optimizationMethod, std::vector<std::string>{"Gradient Descent", "Momentum"},6.0f);
		controls->addNumberField("Epochs", epochs);
		controls->addNumberField("Outer Iterations", iterationsPerEpoch);
		controls->addNumberField("Inner Iterations", iterationsPerStep);
		controls->addNumberField("Learning Rate", learningRateInitial, Float(0.0f), Float(1.0f));
		controls->addNumberField("Learning Delta", learningRateDelta, Float(0.0f), Float(1.0f));
		
	}
	bool NeuralWorker::step() {
		uint64_t iter =iteration;
		bool ret = true;

		if (iter%uint64_t(iterationsPerStep.toInteger())==0&&onUpdate) {
			onUpdate(iter, !ret);
		}
		iteration++;
		return ret;
	}
	NeuralWorker::NeuralWorker(const std::shared_ptr<tgr::NeuralSystem>& system) :
		RecurrentTask([this](uint64_t iteration) {return step();}, 5),sys(system),paused(false) {
		optimizationMethod = 0;
		epochs = Integer(12);
		iterationsPerEpoch = Integer(100);
		iterationsPerStep = Integer(10);
		learningRateInitial = Float(0.8f);
		learningRateDelta = Float(0.5);
	}
}