/*
 * SliceLayer.cpp
 *
 *  Created on: Jul 13, 2017
 *      Author: blake
 */

#include "SliceLayer.h"

namespace tgr {
SliceLayer::SliceLayer(const aly::dim3& in_shape, SliceType slice_type,
		int num_outputs) :
		NeuralLayer("Slice", std::vector<ChannelType>(1, ChannelType::data),
				std::vector<ChannelType>(num_outputs, ChannelType::data)), in_shape(
				in_shape), slice_type(slice_type), num_outputs(num_outputs) {
	set_shape();
}
SliceLayer::SliceLayer(const NeuralLayer &prev_layer, SliceType slice_type,
		int num_outputs) :
		NeuralLayer("Slice", std::vector<ChannelType>(1, ChannelType::data),
				std::vector<ChannelType>(num_outputs, ChannelType::data)), in_shape(
				prev_layer.getOutputDimensions()[0]), slice_type(slice_type), num_outputs(
				num_outputs) {
	set_shape();
}
void SliceLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	switch (slice_type) {
	case SliceType::slice_samples:
		slice_data_forward(*in_data[0], out_data);
		break;
	case SliceType::slice_channels:
		slice_channels_forward(*in_data[0], out_data);
		break;
	default:
		throw std::runtime_error("Not implemented");
	}
}
void SliceLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	CNN_UNREFERENCED_PARAMETER(in_data);
	CNN_UNREFERENCED_PARAMETER(out_data);

	switch (slice_type) {
	case SliceType::slice_samples:
		slice_data_backward(out_grad, *in_grad[0]);
		break;
	case SliceType::slice_channels:
		slice_channels_backward(out_grad, *in_grad[0]);
		break;
	default:
		throw std::runtime_error("Not implemented");
	}
}

void SliceLayer::slice_data_forward(const Tensor &in_data,
		std::vector<Tensor *> &out_data) {
	const Storage *in = &in_data[0];

	for (int i = 0; i < num_outputs; i++) {
		Tensor &out = *out_data[i];

		std::copy(in, in + slice_size[i], &out[0]);

		in += slice_size[i];
	}
}

void SliceLayer::slice_data_backward(std::vector<Tensor *> &out_grad,
		Tensor &in_grad) {
	Storage *in = &in_grad[0];

	for (int i = 0; i < num_outputs; i++) {
		Tensor &out = *out_grad[i];

		std::copy(&out[0], &out[0] + slice_size[i], in);

		in += slice_size[i];
	}
}

void SliceLayer::slice_channels_forward(const Tensor &in_data,
		std::vector<Tensor *> &out_data) {
	int num_samples = static_cast<int>(in_data.size());
	int channel_idx = 0;
	int spatial_dim = in_shape.area();

	for (int i = 0; i < num_outputs; i++) {
		for (int s = 0; s < num_samples; s++) {
			float *out = &(*out_data[i])[s][0];
			const float *in = &in_data[s][0] + channel_idx * spatial_dim;

			std::copy(in, in + slice_size[i] * spatial_dim, out);
		}
		channel_idx += slice_size[i];
	}
}

void SliceLayer::slice_channels_backward(std::vector<Tensor *> &out_grad,
		Tensor &in_grad) {
	int num_samples = static_cast<int>(in_grad.size());
	int channel_idx = 0;
	int spatial_dim = in_shape.area();

	for (int i = 0; i < num_outputs; i++) {
		for (int s = 0; s < num_samples; s++) {
			const float *out = &(*out_grad[i])[s][0];
			float *in = &in_grad[s][0] + channel_idx * spatial_dim;

			std::copy(out, out + slice_size[i] * spatial_dim, in);
		}
		channel_idx += slice_size[i];
	}
}

void SliceLayer::setSampleCount(size_t sample_count) {
	if (slice_type == SliceType::slice_samples) {
		if (num_outputs == 0)
			throw std::runtime_error("num_outputs must be positive integer");

		int sample_per_out = sample_count / num_outputs;

		slice_size.resize(num_outputs, sample_per_out);
		slice_size.back() = sample_count - (sample_per_out * (num_outputs - 1));
	}
	NeuralLayer::setSampleCount(sample_count);
}

void SliceLayer::set_shape() {
	switch (slice_type) {
	case SliceType::slice_samples:
		set_shape_data();
		break;
	case SliceType::slice_channels:
		set_shape_channels();
		break;
	default:
		throw std::runtime_error("not implemented");
	}
}
void SliceLayer::set_shape_data() {
	out_shapes.resize(num_outputs, in_shape);
}
void SliceLayer::set_shape_channels() {
	int channel_per_out = in_shape.z / num_outputs;
	out_shapes.resize(num_outputs);
	for (int i = 0; i < num_outputs; i++) {
		int ch = channel_per_out;

		if (i == num_outputs - 1) {
			assert(in_shape.z >= i * channel_per_out);
			ch = in_shape.z - i * channel_per_out;
		}
		slice_size.push_back(ch);
		out_shapes[i] = aly::dim3(in_shape.x, in_shape.y, ch);
	}
}

}
