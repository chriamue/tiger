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

using namespace aly;
namespace tgr {
	NeuralListener::~NeuralListener() {

	}



	  /**
	   * train on one minibatch
	   *
	   * @param size is the number of data points to use in this batch
	   */
	  template <typename E, typename Optimizer>
	  void train_once(Optimizer &optimizer,
	                  const tensor_t *in,
	                  const tensor_t *t,
	                  int size,
	                  const int nbThreads,
	                  const tensor_t *t_cost) {
	    if (size == 1) {
	      bprop<E>(fprop(in[0]), t[0], t_cost ? t_cost[0] : tensor_t());
	      net_.update_weights(&optimizer, 1);
	    } else {
	      train_onebatch<E>(optimizer, in, t, size, nbThreads, t_cost);
	    }
	  }
	  /**
	   * trains on one minibatch, i.e. runs forward and backward propagation to
	   * calculate
	   * the gradient of the loss function with respect to the network parameters
	   * (weights),
	   * then calls the optimizer algorithm to update the weights
	   *
	   * @param batch_size the number of data points to use in this batch
	   */
	  template <typename E, typename Optimizer>
	  void train_onebatch(Optimizer &optimizer,
	                      const tensor_t *in,
	                      const tensor_t *t,
	                      int batch_size,
	                      const int num_tasks,
	                      const tensor_t *t_cost) {
	    CNN_UNREFERENCED_PARAMETER(num_tasks);
	    std::copy(&in[0], &in[0] + batch_size, &in_batch_[0]);
	    std::copy(&t[0], &t[0] + batch_size, &t_batch_[0]);
	    std::vector<tensor_t> t_cost_batch =
	      t_cost ? std::vector<tensor_t>(&t_cost[0], &t_cost[0] + batch_size)
	             : std::vector<tensor_t>();

	    bprop<E>(fprop(in_batch_), t_batch_, t_cost_batch);
	    net_.update_weights(&optimizer, batch_size);
	  }

	  vec_t fprop(const vec_t &in) {
	    if (in.size() != (size_t)in_data_size()) data_mismatch(**net_.begin(), in);
	#if 0
	        return fprop(std::vector<vec_t>{ in })[0];
	#else
	    // a workaround to reduce memory consumption by skipping wrapper
	    // function
	    std::vector<tensor_t> a(1);
	    a[0].emplace_back(in);
	    return fprop(a)[0][0];
	#endif
	  }

	 bool fit(NeuralOptimizer& optimizer,
	          const std::vector<tensor_t> &inputs,
	          const std::vector<tensor_t> &desired_outputs,
	          size_t batch_size,
	          int epoch,
	          std::function<void()>& on_batch_enumerate,
			  std::function<void()>& on_epoch_enumerate,
	          const bool reset_weights            = false,
	          const int n_threads                 = CNN_TASK_SIZE,
	          const std::vector<tensor_t> &t_cost = std::vector<tensor_t>()) {
	   // check_training_data(in, t);
	   check_target_cost_matrix(desired_outputs, t_cost);
	   set_netphase(net_phase::train);
	   setup(reset_weights);
	   for (auto n : net_) n->set_parallelize(true);
	   optimizer.reset();
	   stop_training_ = false;
	   in_batch_.resize(batch_size);
	   t_batch_.resize(batch_size);
	   for (int iter = 0; iter < epoch && !stop_training_; iter++) {
	     for (size_t i = 0; i < inputs.size() && !stop_training_;
	          i += batch_size) {
	       train_once<Error>(
	         optimizer, &inputs[i], &desired_outputs[i],
	         static_cast<int>(std::min(batch_size, inputs.size() - i)), n_threads,
	         get_target_cost_sample_pointer(t_cost, i));
	       on_batch_enumerate();

	       /* if (i % 100 == 0 && layers_.is_exploded()) {
	         std::cout << "[Warning]Detected infinite value in weight. stop
	       learning." << std::endl;
	           return false;
	       } */
	     }
	     on_epoch_enumerate();
	   }
	   set_netphase(net_phase::test);
	   return true;
	 }
	}

	bool NeuralRuntime::init() {
		lastResidual = 1E30f;
		for (NeuralLayerPtr layer : sys->getLayers()) {
			layer->getGraph()->points.clear();
		}
		/*
		switch (optimizationMethod) {
			case 0:
				sys->setOptimizer(opt = std::shared_ptr<NeuralOptimization>(new GradientDescentOptimizer(learningRateInitial.toFloat(), weightDecay.toFloat())));
				break;
			case 1:
				sys->setOptimizer(opt = std::shared_ptr<NeuralOptimization>(new MomentumOptimizer(learningRateInitial.toFloat(),weightDecay.toFloat(),momentum.toFloat())));
				break;
		}
		sampleIndexes.clear();
		for (int n = lowerSample.toInteger(); n <= upperSample.toInteger(); n++) {
			sampleIndexes.push_back(n);
		}
		sys->initializeWeights(0.0f,1.0f);
		sys->updateKnowledge();
		*/
		cache->clear();
		iteration = 0;
		NeuralKnowledge& k = sys->getKnowledge();
		k.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "tiger" <<std::setw(5)<<std::setfill('0')<< iteration << ".bin");
		k.setName("tiger");
		cache->set(iteration, k);

		return true;
	}
	void NeuralRuntime::cleanup(){
	}
	void NeuralRuntime::setSampleRange(int mn, int mx) {
		minSample.setValue(mn);
		maxSample.setValue(mx);
		batchSize.setValue(std::min(mx-mn+1, batchSize.toInteger()));
	}
	void NeuralRuntime::setSelectedSamples(int mn, int mx) {
		lowerSample.setValue(mn);
		upperSample.setValue(mx);
	}
	void NeuralRuntime::setup(const aly::ParameterPanePtr& controls) {
		controls->addGroup("Training", true);
		controls->addSelectionField("Optimizer", optimizationMethod, std::vector<std::string>{"Gradient Descent", "Momentum"},6.0f);
		controls->addNumberField("Epochs", iterationsPerEpoch);
		controls->addRangeField("Samples", lowerSample, upperSample, minSample, maxSample);
		controls->addNumberField("Batch Size", batchSize, Integer(0), Integer((maxSample.toInteger()- minSample.toInteger())+1));
		controls->addNumberField("Learning Rate", learningRateInitial, Float(0.0f), Float(1.0f));
		controls->addNumberField("Attenuation", learningRateDelta, Float(0.0f), Float(1.0f));
		controls->addNumberField("Weight Decay", weightDecay, Float(0.0f), Float(1.0f));
		controls->addNumberField("Momentum", momentum, Float(0.0f), Float(1.0f));
	}
	bool NeuralRuntime::step() {
		static std::random_device rd;
		int iter =iteration;
		bool ret = true;
		double res = 0;
		//sys->reset();
		int B = std::min(batchSize.toInteger(),(int)sampleIndexes.size());
		/*
		if (iteration == 0) {
			opt->setLearningRate(opt->getLearningRate() / B);
		}
		if (iter%iterationsPerStep.toInteger() == 0) {
			std::shuffle(sampleIndexes.begin(), sampleIndexes.end(), rd);
		}
		for(int b=0;b<B;b++){
			int idx = sampleIndexes[(b+iter*B)%sampleIndexes.size()];
			if (inputSampler)inputSampler(sys->getInput(), idx);
			sys->evaluate();
			if (outputSampler) {
				outputSampler(outputData, idx);
				double err= sys->accumulate(outputData);
				res += err;
				//std::cout << "Evaluate ["<<idx<<"] Error=" << err <<" "<< std::endl;
			}
			sys->backpropagate();
		}
		std::cout << iter<<") Residual Error=" << res << " " << std::endl;
		double delta = std::abs(lastResidual - res);
		if (delta < 1E-5f) {
			opt->setLearningRate(opt->getLearningRate()*learningRateDelta.toFloat());
			std::cout << "Learning Rate=" << opt->getLearningRate()*B << std::endl;
		}
		lastResidual = res;
		sys->optimize();
		for (NeuralLayerPtr layer : sys->getLayers()) {
			layer->getGraph()->points.push_back(float2(float(iter),float(layer->getResidual())));
		}
		if (delta<1E-7f||iteration>=(int)getMaxIteration()) {
			ret = false;
		}
		if (onUpdate) {
			onUpdate(iter, !ret);
		}
		*/
		iteration++;
		sys->updateKnowledge();
		NeuralKnowledge& k = sys->getKnowledge();
		k.setFile(MakeString() << GetDesktopDirectory() << ALY_PATH_SEPARATOR << "tiger" << std::setw(5) << std::setfill('0') << iteration << ".bin");
		k.setName("tiger");
		cache->set(iteration, k);
		return ret;
	}
	NeuralRuntime::NeuralRuntime(const std::shared_ptr<tgr::NeuralSystem>& system) :
		RecurrentTask([this](uint64_t iteration) {return step();}, 5),paused(false),sys(system) {
		optimizationMethod = 1;
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
		cache.reset(new NeuralCache());
	}
}
