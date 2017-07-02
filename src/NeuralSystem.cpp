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
void NeuralSystem::reorder_for_layerwise_processing(
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
	reorder_for_layerwise_processing(out_grad, reordered_grad);
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
	reorder_for_layerwise_processing(in_data, reordered_data);
	assert(reordered_data.size() == input_data_channel_count);
	for (size_t channel_index = 0; channel_index < input_data_channel_count;
			channel_index++) {
		inputLayers[channel_index]->setInputData({reordered_data[channel_index]});
	}

	for (auto l : layers) {
		l->forward();
	}
	return mergeOutputs();
}

void NeuralSystem::build(const std::vector<NeuralLayerPtr> &input,const std::vector<NeuralLayerPtr> &output) {
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
		if(curr->isRoot()){
			roots.push_back(curr);
		}
		std::cout<<"Current "<<curr->getName()<<" ROOT? "<<curr->isRoot()<<" "<<curr->getInputSignals().size()<<" "<<curr->getOutputSignals().size()<<std::endl;
		std::vector<NeuralLayerPtr> next = curr->getOutputLayers();
		for (size_t i = 0; i < next.size(); i++) {
			if (!next[i])
				continue;
			// remove edge between next[i] and current
			if (removed_edge.find(next[i]) == removed_edge.end()) {
				removed_edge[next[i]] = std::vector<uint8_t>(next[i]->getInputLayers().size(), 0);
			}
			std::vector<uint8_t> &removed = removed_edge[next[i]];
			int idx=0;
			std::vector<NeuralLayer*> nodes=next[i]->getInputLayers();
			for(int n=0;n<(int)nodes.size();n++){
				if(nodes[n]==curr.get()){
					removed[n] = 1;
				}
			}
			if (std::all_of(removed.begin(), removed.end(),[](uint8_t x) {return x == 1;})) {
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
