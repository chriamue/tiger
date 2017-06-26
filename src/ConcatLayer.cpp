/*
 * ConcatLayer.cpp
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#include "ConcatLayer.h"
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::core;
namespace tgr {
ConcatLayer::ConcatLayer(const std::vector<aly::dim3> &in_shapes) :
		NeuralLayer("Concat",
				std::vector<ChannelType>(in_shapes.size(), ChannelType::data), {
						ChannelType::data }), in_shapes_(in_shapes) {
	set_outshape();
}
ConcatLayer::ConcatLayer(int num_args, int ndim) :
		NeuralLayer("Concat",
				std::vector<ChannelType>(num_args, ChannelType::data), {
						ChannelType::data }), in_shapes_(
				std::vector<aly::dim3>(num_args, aly::dim3(ndim, 1, 1))) {
	set_outshape();
}
void ConcatLayer::set_outshape() {
	out_shape_ = in_shapes_.front();
	for (size_t i = 1; i < in_shapes_.size(); i++) {
		if (in_shapes_[i].x * in_shapes_[i].y != out_shape_.x * out_shape_.y)
			throw nn_error(
					"each input shapes to concat must have same WxH size");
		out_shape_.z += in_shapes_[i].z;
	}
}
std::vector<aly::dim3> ConcatLayer::getInputDimensions() const {
	return in_shapes_;
}
std::vector<aly::dim3> ConcatLayer::getOutputDimensions() const {
	return {out_shape_};
}
void ConcatLayer::forwardPropagation(const std::vector<Tensor *> &in_data,
		std::vector<Tensor *> &out_data) {
	int num_samples = static_cast<int>((*out_data[0]).size());
	tiny_dnn::for_i(num_samples, [&](size_t s) {
		float_t *outs = &(*out_data[0])[s][0];

		for (int i = 0; i < in_shapes_.size(); i++) {
			const float_t *ins = &(*in_data[i])[s][0];
			int dim = in_shapes_[i].size();
			outs = std::copy(ins, ins + dim, outs);
		}
	});
}
void ConcatLayer::backwardPropagation(const std::vector<Tensor *> &in_data,
		const std::vector<Tensor *> &out_data, std::vector<Tensor *> &out_grad,
		std::vector<Tensor *> &in_grad) {
	CNN_UNREFERENCED_PARAMETER(in_data);
	CNN_UNREFERENCED_PARAMETER(out_data);
	size_t num_samples = (*out_grad[0]).size();
	tiny_dnn::for_i(num_samples, [&](size_t s) {
		const float_t *outs = &(*out_grad[0])[s][0];

		for (int i = 0; i < in_shapes_.size(); i++) {
			int dim = in_shapes_[i].size();
			float_t *ins = &(*in_grad[i])[s][0];
			std::copy(outs, outs + dim, ins);
			outs += dim;
		}
	});
}
}

