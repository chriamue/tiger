/*
 * PartialConnectedLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "PartialConnectedLayer.h"
#include "tiny_dnn/util/util.h"
namespace tgr {
PartialConnectedLayer::PartialConnectedLayer(const std::string& name,int in_dim, int out_dim,
		size_t weight_dim, size_t bias_dim,float scale_factor) :
		NeuralLayer(name, ChannelOrder(bias_dim > 0), {
				ChannelType::data }), weight2io_(weight_dim), out2wi_(out_dim), in2wo_(
				in_dim), bias2out_(bias_dim), out2bias_(out_dim), scale_factor_(
				scale_factor) {
}
size_t PartialConnectedLayer::param_size() const {
	size_t total_param = 0;
	for (auto w : weight2io_)
		if (w.size() > 0)
			total_param++;
	for (auto b : bias2out_)
		if (b.size() > 0)
			total_param++;
	return total_param;
}

int PartialConnectedLayer::getFanInSize() const {
	return tiny_dnn::max_size(out2wi_);
}

int PartialConnectedLayer::getFanOutSize() const {
	return tiny_dnn::max_size(in2wo_);
}

void PartialConnectedLayer::connect_weight(int input_index, int output_index,
		int weight_index) {
	weight2io_[weight_index].emplace_back(input_index, output_index);
	out2wi_[output_index].emplace_back(weight_index, input_index);
	in2wo_[input_index].emplace_back(weight_index, output_index);
}

void PartialConnectedLayer::connect_bias(int bias_index, int output_index) {
	out2bias_[output_index] = bias_index;
	bias2out_[bias_index].push_back(output_index);
}

void PartialConnectedLayer::forwardPropagation(
		const std::vector<Tensor *> &in_data, std::vector<Tensor *> &out_data) {
	const Tensor &in = *in_data[0];
	const Storage &W = (*in_data[1])[0];
	const Storage &b = (*in_data[2])[0];
	Tensor &out = *out_data[0];

	// @todo revise the parallelism strategy
	for (int sample = 0, sample_count = static_cast<int>(in.size());
			sample < sample_count; ++sample) {
		Storage &out_sample = out[sample];
		tiny_dnn::for_i(out2wi_.size(), [&](size_t i) {
			const wi_connections &connections = out2wi_[i];

			float_t &out_element = out_sample[i];

			out_element = float_t {0};

			for (auto connection : connections)
			out_element += W[connection.first] * in[sample][connection.second];

			out_element *= scale_factor_;
			out_element += b[out2bias_[i]];
		});
	}
}

void PartialConnectedLayer::backwardPropagation(
		const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	const Tensor &prev_out = *in_data[0];
	const Storage &W = (*in_data[1])[0];
	Storage &dW = (*in_grad[1])[0];
	Storage &db = (*in_grad[2])[0];
	Tensor &prev_delta = *in_grad[0];
	Tensor &curr_delta = *out_grad[0];

	// @todo revise the parallelism strategy
	for (int sample = 0, sample_count = static_cast<int>(prev_out.size());
			sample < sample_count; ++sample) {
		tiny_dnn::for_i(in2wo_.size(),
				[&](size_t i) {
					const wo_connections &connections = in2wo_[i];
					float_t delta {0};

					for (auto connection : connections)
					delta += W[connection.first] * curr_delta[sample][connection.second];

					prev_delta[sample][i] = delta * scale_factor_;
				});

		tiny_dnn::for_i(weight2io_.size(), [&](size_t i) {
			const io_connections &connections = weight2io_[i];
			float_t diff {0};

			for (auto connection : connections)
			diff += prev_out[sample][connection.first] *
			curr_delta[sample][connection.second];

			dW[i] += diff * scale_factor_;
		});

		for (size_t i = 0; i < bias2out_.size(); i++) {
			const std::vector<int> &outs = bias2out_[i];
			float_t diff { 0 };

			for (auto o : outs)
				diff += curr_delta[sample][o];

			db[i] += diff;
		}
	}
}
}

