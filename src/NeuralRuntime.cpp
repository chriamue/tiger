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
#include <random>
#include <omp.h>
using namespace aly;
namespace tgr {
NeuralListener::~NeuralListener() {

}

/**
 * train on one minibatch
 *
 * @param size is the number of data points to use in this batch
 */
void NeuralRuntime::trainOnce(NeuralOptimizer &optimizer,
		const NeuralLossFunction& loss, const Tensor* in, const Tensor* t,
		int size, const int nbThreads, const Tensor *t_cost) {
	if (size == 1) {
		sys->bprop(loss, sys->fprop(in[0]), t[0],t_cost ? t_cost[0] : Tensor());
		sys->updateWeights(optimizer, 1);
	} else {
		trainOneBatch(optimizer, loss, in, t, size, nbThreads, t_cost);
	}
}
/**
 * trains on one minibatch, i.e. runs forward and backward propagation to
 * calculate
 * the gradient of the loss function with respect to the network parameters
 * (weights),
 * then calls the optimizer algorithm to update the weights
 * @param batch_size the number of data points to use in this batch
 */
void NeuralRuntime::trainOneBatch(NeuralOptimizer &optimizer,
		const NeuralLossFunction& loss, const Tensor *in, const Tensor *t,
		int batch_size, const int num_tasks, const Tensor *t_cost) {
	CNN_UNREFERENCED_PARAMETER(num_tasks);
	in_batch.resize(batch_size);
	t_batch.resize(batch_size);
	std::copy(&in[0], &in[0] + batch_size, &in_batch[0]);
	std::copy(&t[0], &t[0] + batch_size, &t_batch[0]);
	std::vector<Tensor> t_cost_batch =
			t_cost ?
					std::vector<Tensor>(&t_cost[0], &t_cost[0] + batch_size) :
					std::vector<Tensor>();
	//Perform forward and backward pass on in_batch
	sys->bprop(loss, sys->fprop(in_batch), t_batch, t_cost_batch);
	sys->updateWeights(optimizer, batch_size);
}
float NeuralRuntime::getLoss(const NeuralLossFunction& loss) {
	return sys->getLoss(loss, inputs, desiredOutputs);
}

void NeuralRuntime::setData(const std::vector<Tensor>& inputs,
		std::vector<Tensor>& desiredOutputs,
		const std::vector<Tensor> &t_cost) {
	this->inputs = inputs;
	this->desiredOutputs = desiredOutputs;
	this->t_costs = t_cost;
}
void NeuralRuntime::setData(const std::vector<Storage> &inputs,
		const std::vector<int> &class_labels,
		const std::vector<Storage>& t_cost) {
	sys->normalize(inputs, this->inputs);
	sys->normalize(class_labels, this->desiredOutputs);
	if (!t_cost.empty())
		sys->normalize(t_cost, this->t_costs);
}
void NeuralRuntime::setData(const std::vector<Tensor> &inputs,
		const std::vector<int> &class_labels,
		const std::vector<Storage>& t_cost) {
	this->inputs = inputs;
	sys->normalize(class_labels, this->desiredOutputs);
	if (!t_cost.empty())
		sys->normalize(t_cost, this->t_costs);
}
const Tensor* NeuralRuntime::get_target_cost_sample_pointer(
		const std::vector<Tensor> &t_cost, size_t i) {
	if (!t_cost.empty()) {
		assert(i < t_cost.size());
		return &(t_cost[i]);
	} else {
		return nullptr;
	}
}
bool NeuralRuntime::init(bool reset_weights) {
	lastResidual = 1E30f;
	if (optimizationMethod >= 0) {
		switch (optimizationMethod) {
		case 0: {
			GradientDescentOptimizer g = GradientDescentOptimizer();
			g.alpha = learningRateDelta.toFloat();
			g.lambda = weightDecay.toFloat();
			optimizer = g;
		}
			break;
		case 1: {
			MomentumOptimizer m = MomentumOptimizer();
			m.mu = momentum.toFloat();
			m.alpha = learningRateDelta.toFloat();
			m.lambda = weightDecay.toFloat();
			optimizer = m;
		}
			break;
		case 2: {
			AdamOptimizer ad = AdamOptimizer();
			ad.alpha = learningRateDelta.toFloat();
			optimizer = ad;
		}
			break;
		case 3: {
			AdagradOptimizer ag = AdagradOptimizer();
			ag.alpha = learningRateDelta.toFloat();
			optimizer = ag;
		}
			break;
		case 4: {
			RMSpropOptimizer ro = RMSpropOptimizer();
			ro.alpha = learningRateDelta.toFloat();
			optimizer = ro;
		}
			break;
		default:
			throw std::runtime_error("No optimizer specified.");
			break;
		}
	}
	if (lossFunction >= 0) {
		switch (lossFunction) {
		case 0:
			loss = MSELossFunction();
			break;
		case 1:
			loss = AbsoluteLossFunction();
			break;
		case 2:
			loss = AbsoluteEpsLossFunction();
			break;
		case 3:
			loss = CrossEntropyLossFunction();
			break;
		case 4:
			loss = CrossEntropyMultiClassLossFunction();
			break;
		default:
			throw std::runtime_error("No loss function specified.");
			break;
		}
	}
	sys->getGraph()->points.clear();
	sys->setPhase(NetPhase::Train);
	sys->setup(reset_weights);
	for (auto n : sys->getLayers()) {
		n->setParallelize(true);
	}
	optimizer.reset();
	running = true;
	int batch_size = batchSize.toInteger();
	in_batch.resize(batch_size);
	t_batch.resize(batch_size);
	iteration = 0;
	return true;
}
void NeuralRuntime::cleanup() {
	sys->setPhase(NetPhase::Test);
}
void NeuralRuntime::setSampleRange(int mn, int mx) {
	minSample.setValue(mn);
	maxSample.setValue(mx);
	batchSize.setValue(std::min(mx - mn + 1, batchSize.toInteger()));
}
void NeuralRuntime::setSelectedSamples(int mn, int mx) {
	lowerSample.setValue(mn);
	upperSample.setValue(mx);
}
void NeuralRuntime::setup(const aly::ParameterPanePtr& controls) {
	controls->addGroup("Training", true);
	optimizationMethod = 1;
	lossFunction = 0;
	controls->addSelectionField("Optimizer", optimizationMethod,
			std::vector<std::string> { "Gradient Descent", "Momentum", "Adam",
					"Adagrad", "RMSprop" }, 6.0f);
	controls->addSelectionField("Error Metric", lossFunction,
			std::vector<std::string> { "MSE", "Absolute", "Absolute Epsilon",
					"Cross Entropy", "Cross Class Entropy" }, 6.0f);
	controls->addNumberField("Epochs", iterationsPerEpoch);
	controls->addRangeField("Samples", lowerSample, upperSample, minSample,
			maxSample);
	controls->addNumberField("Batch Size", batchSize, Integer(0),
			Integer((maxSample.toInteger() - minSample.toInteger()) + 1));
	controls->addNumberField("Learning Rate", learningRateInitial, Float(0.0f),
			Float(1.0f));
	controls->addNumberField("Attenuation", learningRateDelta, Float(0.0f),
			Float(1.0f));
	controls->addNumberField("Weight Decay", weightDecay, Float(0.0f),
			Float(1.0f));
	controls->addNumberField("Momentum", momentum, Float(0.0f), Float(1.0f));
}
bool NeuralRuntime::step() {
	static std::random_device rd;
	int iter = iteration;
	bool ret = true;
	double res = 0;
	int batch_size = batchSize.toInteger();
	for (size_t i = lowerSample.toInteger(); i <= upperSample.toInteger() && running; i += batch_size) {
		int sz = std::min(batch_size,(int) (upperSample.toInteger() + 1 - i));
		if (sz > 0) {
			trainOnce(optimizer, loss, &inputs[i], &desiredOutputs[i], sz, threads, get_target_cost_sample_pointer(t_costs, i));
			if (onBatchEnumerate)
				onBatchEnumerate();
		}
	}
	std::cout<<"Evaluate"<<std::endl;
	float err = getLoss(loss);
	sys->getGraph()->points.push_back(float2(iteration, err));
	std::cout << "Error Loss " << err << std::endl;
	ret=(iter<getMaxIteration()-1);
	if (onEpochEnumerate)
		onEpochEnumerate();
	if (onUpdate) {
		onUpdate(iter,!ret);
	}
	iteration++;
	//sys->updateKnowledge();
	//NeuralKnowledge& k = sys->getKnowledge();
	//k.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR<< "tiger" << std::setw(5) << std::setfill('0') << iteration << ".bin");
	//k.setName("tiger");
	//cache->set(iteration, k);
	std::cout<<iteration<<"/"<<getMaxIteration()<<" "<<ret<<std::endl;
	return ret;
}
NeuralRuntime::NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system) :
		RecurrentTask([this](uint64_t iteration) {return step();}, 5), paused(
				false), sys(system) {
	optimizationMethod = -1;
	iterationsPerEpoch = Integer(200);
	iterationsPerStep = Integer(10);
	batchSize = Integer(32);
	lowerSample = Integer(0);
	upperSample = Integer(0);
	minSample = Integer(0);
	maxSample = Integer(0);
	learningRateInitial = Float(0.999f);
	weightDecay = Float(0.0f);
	momentum = Float(0.9f);
	learningRateDelta = Float(0.9f);
	threads = omp_get_max_threads();
	cache.reset(new NeuralCache());
}
}
