/*
 * ConvolutiondnnLayer.cpp
 *
 *  Created on: Jun 24, 2017
 *      Author: blake
 */

#include "ConvolutionLayer.h"
using namespace tiny_dnn;
using namespace tiny_dnn::core;
namespace tgr {
ConvolutionLayer::ConvolutionLayer(int in_width, int in_height,
		int window_width, int window_height, int in_channels, int out_channels,
		const core::connection_table& connection_table, Padding pad_type, bool has_bias, int w_stride,
		int h_stride , BackendType backend_type ) :
		NeuralLayer("Convolution", ChannelOrder(has_bias),
				{ ChannelType::data }), dnnLayer(in_width, in_height, window_width,
				in_channels, out_channels, connection_table,
				static_cast<padding>(pad_type), has_bias, w_stride, h_stride,
				static_cast<backend_t>(backend_type)) {
}
std::vector<aly::dim3> ConvolutionLayer::getInputDimensions() const {
	return Convert(dnnLayer.in_shape());
}
std::vector<aly::dim3> ConvolutionLayer::getOutputDimensions() const {
	return Convert(dnnLayer.out_shape());
}
void ConvolutionLayer::setInputShape(const aly::int3& in_shape) {
	dnnLayer.set_in_shape(shape3d(in_shape.x, in_shape.y, in_shape.z));
}
void ConvolutionLayer::post() {
	dnnLayer.post_update();
}
void ConvolutionLayer::forwardPropagation(const std::vector<Tensor*>&in_data,
		std::vector<Tensor*> &out_data) {
	dnnLayer.forward_propagation(in_data, out_data);
}
void ConvolutionLayer::backwardPropagation(const std::vector<Tensor*> &in_data,
		const std::vector<Tensor*> &out_data, std::vector<Tensor*> &out_grad,
		std::vector<Tensor*> &in_grad) {
	dnnLayer.back_propagation(in_data, out_data, out_grad, in_grad);
}
void ConvolutionLayer::setSampleCount(size_t sample_count) {
	dnnLayer.set_sample_count(sample_count);
}
int ConvolutionLayer::getFanInSize() const {
	return dnnLayer.fan_in_size();
}
int ConvolutionLayer::getFanOutSize() const {
	return dnnLayer.fan_out_size();
}

}
