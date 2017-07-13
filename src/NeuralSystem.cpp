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
#include "NeuralSystem.h"
#include "NeuralFlowPane.h"

using namespace aly;
namespace tgr {

NeuralSystem::NeuralSystem(const std::string& name,
		const std::shared_ptr<aly::NeuralFlowPane>& pane) :
		name(name), initialized(false), flowPane(pane) {

}

NeuralKnowledge& NeuralSystem::updateKnowledge() {
	knowledge.set(*this);
	return knowledge;
}
Storage NeuralSystem::predict(const Storage &in) {
	std::vector<Tensor> a(1);
	a[0].emplace_back(in);
	return forward(a)[0][0];
}
Tensor NeuralSystem::predict(const Tensor &in) {
	return forward({in})[0];
}
std::vector<Tensor> NeuralSystem::predict(const std::vector<Tensor>& in) {
	return forward(in);
}
void NeuralSystem::setup(bool reset_weight) {
	for (auto l : layers) {
		l->setup(reset_weight);
	}
}

void NeuralSystem::clearGradients() {
	for (auto l : layers) {
		l->clearGradients();
	}
}
// transform indexing so that it's more suitable for per-layer operations
// input:  [sample][channel][feature]
// output: [channel][sample][feature]
void NeuralSystem::reorderForLayerwiseProcessing(
		const std::vector<Tensor> &input,
		std::vector<std::vector<const Storage *>> &output) {
	size_t sample_count = input.size();
	size_t channel_count = input[0].size();
	output.resize(channel_count);
	for (size_t i = 0; i < channel_count; ++i) {
		output[i].resize(sample_count);
	}
	for (size_t sample = 0; sample < sample_count; ++sample) {
		assert(input[sample].size() == channel_count);
		for (size_t channel = 0; channel < channel_count; ++channel) {
			output[channel][sample] = &input[sample][channel];
		}
	}
}

void NeuralSystem::backward(const std::vector<Tensor> &out_grad) {
	size_t output_channel_count = out_grad[0].size();
	if (output_channel_count != outputLayers.size()) {
		throw std::runtime_error("input size mismatch");
	}
	std::vector<std::vector<const Storage *>> reordered_grad;
	reorderForLayerwiseProcessing(out_grad, reordered_grad);
	assert(reordered_grad.size() == output_channel_count);
	for (size_t i = 0; i < output_channel_count; i++) {
		outputLayers[i]->setOutputGradients( { reordered_grad[i] });
	}
	for (auto l = layers.rbegin(); l != layers.rend(); l++) {
		(*l)->backward();
	}
}
std::vector<Tensor> NeuralSystem::mergeOutputs() {
	std::vector<Tensor> merged;
	std::vector<Tensor*> out;
	size_t output_channel_count = outputLayers.size();
	for (size_t output_channel = 0; output_channel < output_channel_count;
			++output_channel) {
		outputLayers[output_channel]->getOutput(out);
		size_t sample_count = out[0]->size();
		if (output_channel == 0) {
			assert(merged.empty());
			merged.resize(sample_count, Tensor(output_channel_count));
		}

		assert(merged.size() == sample_count);

		for (size_t sample = 0; sample < sample_count; ++sample) {
			merged[sample][output_channel] = (*out[0])[sample];
		}
	}
	return merged;
}
std::vector<Tensor> NeuralSystem::forward(const std::vector<Tensor> &in_data) {
	size_t input_data_channel_count = in_data[0].size();
	if (input_data_channel_count != inputLayers.size()) {
		throw std::runtime_error("input size mismatch");
	}
	std::vector<std::vector<const Storage *>> reordered_data;
	reorderForLayerwiseProcessing(in_data, reordered_data);
	assert(reordered_data.size() == input_data_channel_count);
	for (size_t channel_index = 0; channel_index < input_data_channel_count; channel_index++) {
		inputLayers[channel_index]->setInputData({reordered_data[channel_index]});
	}
	for (auto l : layers) {
		l->forward();
	}
	return mergeOutputs();
}
void NeuralSystem::evaluate() {
	for (auto l : layers) {
		l->forward();
	}
}
size_t NeuralSystem::getInputDataSize() const {
	return layers.front()->getInputDataSize();
}
size_t NeuralSystem::getOutputDataSize() const {
	return layers.back()->getOutputDataSize();
}

void NeuralSystem::label2vec(const int *t, int num,
		std::vector<Storage> &vec) const {
	size_t outdim = getOutputDataSize();
	vec.reserve(num);
	for (int i = 0; i < num; i++) {
		assert(t[i] < outdim);
		vec.emplace_back(outdim, getTargetValueMin());
		vec.back()[t[i]] = getTargetValueMax();
	}
}
void NeuralSystem::label2vec(const std::vector<int> &labels,
		std::vector<Storage> &vec) const {
	return label2vec(&labels[0], static_cast<int>(labels.size()), vec);
}
float NeuralSystem::getTargetValueMin() const {
	return layers.back()->getOutputRange().x;
}
float NeuralSystem::getTargetValueMax() const {
	return layers.back()->getOutputRange().y;
}
void NeuralSystem::normalize(const std::vector<Tensor> &inputs,
		std::vector<Tensor> &normalized) {
	normalized = inputs;
}
void NeuralSystem::normalize(const std::vector<Storage> &inputs,
		std::vector<Tensor> &normalized) {
	normalized.reserve(inputs.size());
	for (size_t i = 0; i < inputs.size(); i++)
		normalized.emplace_back(Tensor { inputs[i] });
}

void NeuralSystem::normalize(const std::vector<int> &inputs,
		std::vector<Tensor> &normalized) {
	std::vector<Storage> vec;
	normalized.reserve(inputs.size());
	label2vec(inputs, vec);
	normalize(vec, normalized);
}
void NeuralSystem::setPhase(NetPhase phase) {
	for (auto n : layers) {
		n->setContext(phase);
	}
}
std::vector<Storage> NeuralSystem::test(const std::vector<Storage> &in) {
	std::vector<Storage> test_result(in.size());
	setPhase(NetPhase::Test);
	for (size_t i = 0; i < in.size(); i++) {
		test_result[i] = predict(in[i]);
	}
	return test_result;
}

float NeuralSystem::getLoss(const NeuralLossFunction& loss,
		const std::vector<Tensor> &in, const std::vector<Tensor> &t) {
	float sum_loss = float(0);
	std::vector<Tensor> in_tensor;
	normalize(in, in_tensor);
	for (size_t i = 0; i < in.size(); i++) {
		const Tensor predicted = predict(in_tensor[i]);
		for (size_t j = 0; j < predicted.size(); j++) {
			sum_loss += loss.f(predicted[j], t[i][j]);
		}
	}
	return sum_loss;
}
float NeuralSystem::getLoss(const NeuralLossFunction& loss,
		const std::vector<Storage> &in, const std::vector<Storage> &t) {
	float sum_loss = float(0);
	for (size_t i = 0; i < in.size(); i++) {
		const Storage predicted = predict(in[i]);
		sum_loss += loss.f(predicted, t[i]);
	}
	return sum_loss;
}

float NeuralSystem::getLoss(const NeuralLossFunction& loss,
		const std::vector<Storage> &in, const std::vector<Tensor> &t) {
	float sum_loss = float(0);
	std::vector<Tensor> in_tensor;
	normalize(in, in_tensor);
	for (size_t i = 0; i < in.size(); i++) {
		const Tensor predicted = predict(in_tensor[i]);
		for (size_t j = 0; j < predicted.size(); j++) {
			sum_loss += loss.f(predicted[j], t[i][j]);
		}
	}
	return sum_loss;
}
float NeuralSystem::getLoss(const NeuralLossFunction& loss,
		const std::vector<int> &in, const std::vector<Tensor> &t) {
	float sum_loss = float(0);
	std::vector<Tensor> in_tensor;
	normalize(in, in_tensor);
	for (size_t i = 0; i < in.size(); i++) {
		const Tensor predicted = predict(in_tensor[i]);
		for (size_t j = 0; j < predicted.size(); j++) {
			sum_loss += loss.f(predicted[j], t[i][j]);
		}
	}
	return sum_loss;
}

Storage NeuralSystem::fprop(const Storage &in) {
	// a workaround to reduce memory consumption by skipping wrapper
	// function
	std::vector<Tensor> a(1);
	a[0].emplace_back(in);
	return fprop(a)[0][0];
}

// convenience wrapper for the function below
std::vector<Storage> NeuralSystem::fprop(const std::vector<Storage> &in) {
	return fprop(std::vector<Tensor> { in })[0];
}
std::vector<Tensor> NeuralSystem::fprop(const std::vector<Tensor> &in) {
	return forward(in);
}
/**
 * checking gradients calculated by bprop
 * detail information:
 * http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
 **/
bool NeuralSystem::calculateDelta(const NeuralLossFunction& loss,
		const std::vector<Tensor> &in, const std::vector<Tensor> &v, Storage &w,
		Tensor &dw, int check_index, double eps) {
	static const float delta = std::sqrt(std::numeric_limits<float>::epsilon());

	assert(in.size() == v.size());

	const int sample_count = static_cast<int>(in.size());

	assert(sample_count > 0);

	// at the moment, channel count must be 1
	assert(in[0].size() == 1);
	assert(v[0].size() == 1);

	// clear previous results, if any
	for (Storage &dw_sample : dw) {
		vectorize::fill(&dw_sample[0], dw_sample.size(), float(0));
	}

	// calculate dw/dE by numeric
	float prev_w = w[check_index];

	float f_p = float(0);
	w[check_index] = prev_w + delta;
	for (int i = 0; i < sample_count; i++) {
		f_p += getLoss(loss, in[i], v[i]);
	}

	float f_m = float(0);
	w[check_index] = prev_w - delta;
	for (int i = 0; i < sample_count; i++) {
		f_m += getLoss(loss, in[i], v[i]);
	}

	float delta_by_numerical = (f_p - f_m) / (float(2) * delta);
	w[check_index] = prev_w;

	// calculate dw/dE by bprop
	bprop(loss, fprop(in), v, std::vector<Tensor>());

	float delta_by_bprop = 0;
	for (int sample = 0; sample < sample_count; ++sample) {
		delta_by_bprop += dw[sample][check_index];
	}
	clearGradients();

	return std::abs(delta_by_bprop - delta_by_numerical) <= eps;
}
// convenience wrapper for the function below
void NeuralSystem::bprop(const NeuralLossFunction& loss,
		const std::vector<Storage> &out, const std::vector<Storage> &t,
		const std::vector<Storage> &t_cost) {
	bprop(loss, std::vector<Tensor> { out }, std::vector<Tensor> { t },
			std::vector<Tensor> { t_cost });
}
void NeuralSystem::bprop(const NeuralLossFunction& loss,
		const std::vector<Tensor> &out, const std::vector<Tensor> &t,
		const std::vector<Tensor> &t_cost) {
	std::vector<Tensor> delta = loss.gradient(out, t, t_cost);
	backward(delta);
}
bool NeuralSystem::gradientCheck(const NeuralLossFunction& loss,
		const std::vector<Tensor> &in, const std::vector<std::vector<int>> &t,
		float eps, GradientCheck mode) {
	assert(in.size() == t.size());

	std::vector<Tensor> v(t.size());
	const int sample_count = static_cast<int>(t.size());
	for (int sample = 0; sample < sample_count; ++sample) {
		label2vec(t[sample], v[sample]);
	}

	for (auto current : layers) {  // ignore first input layer
		if (current->getInputWeights().size() < 2) {
			continue;
		}
		Storage &w = *current->getInputWeights()[0];
		Storage &b = *current->getInputWeights()[1];
		Tensor &dw = (*current->getInputGradient()[0]);
		Tensor &db = (*current->getInputGradient()[1]);
		if (w.empty())
			continue;

		switch (mode) {
		case GradientCheck::All:
			for (size_t i = 0; i < w.size(); i++)
				if (!calculateDelta(loss, in, v, w, dw, i, eps)) {
					return false;
				}
			for (size_t i = 0; i < b.size(); i++)
				if (!calculateDelta(loss, in, v, b, db, i, eps)) {
					return false;
				}
			break;
		case GradientCheck::Random:
			for (size_t i = 0; i < 10; i++)
				if (!calculateDelta(loss, in, v, w, dw,
						RandomUniform(0, w.size() - 1), eps)) {
					return false;
				}
			for (size_t i = 0; i < 10; i++)
				if (!calculateDelta(loss, in, v, b, db,
						RandomUniform(0, b.size() - 1), eps)) {
					return false;
				}
			break;
		default:
			throw std::runtime_error("unknown grad-check type");
		}
	}
	return true;
}

void NeuralSystem::build(const std::vector<NeuralLayerPtr> &input,
		const std::vector<NeuralLayerPtr> &output) {
	std::vector<NeuralLayerPtr> sorted;
	std::vector<NeuralLayerPtr> input_nodes(input.begin(), input.end());
	std::unordered_map<NeuralLayerPtr, std::vector<uint8_t>> removed_edge;
	layers.clear();
	roots.clear();
// topological-sorting
	while (!input_nodes.empty()) {
		sorted.push_back(input_nodes.back());
		input_nodes.pop_back();
		NeuralLayerPtr curr = sorted.back();
		curr->setSystem(this);
		if (curr->isRoot()) {
			roots.push_back(curr);
		}
		std::vector<NeuralLayerPtr> next = curr->getOutputLayers();
		for (size_t i = 0; i < next.size(); i++) {
			if (!next[i])
				continue;
			// remove edge between next[i] and current
			if (removed_edge.find(next[i]) == removed_edge.end()) {
				removed_edge[next[i]] = std::vector<uint8_t>(
						next[i]->getInputLayers().size(), 0);
			}
			std::vector<uint8_t> &removed = removed_edge[next[i]];
			int idx = 0;
			std::vector<NeuralLayer*> nodes = next[i]->getInputLayers();
			for (int n = 0; n < (int) nodes.size(); n++) {
				if (nodes[n] == curr.get()) {
					removed[n] = 1;
				}
			}
			if (std::all_of(removed.begin(), removed.end(),
					[](uint8_t x) {return x == 1;})) {
				input_nodes.push_back(next[i]);
			}
		}
	}
	for (auto &n : sorted) {
		layers.push_back(n);
	}
	inputLayers = input;
	outputLayers = output;
	setup(false);
}
void NeuralSystem::updateWeights(NeuralOptimizer& opt, int batch_size) {
	for (auto l : layers) {
		l->updateWeights(opt, batch_size);
	}
}

void NeuralSystem::initialize() {
	setup(true);
}
void NeuralSystem::initialize(const aly::ExpandTreePtr& tree) {
	TreeItemPtr root = TreeItemPtr(new TreeItem("Neural Layers"));
	tree->addItem(root);
	root->setExpanded(true);
	for (NeuralLayerPtr n : roots) {
		n->initialize(tree, root);
	}
}

}
