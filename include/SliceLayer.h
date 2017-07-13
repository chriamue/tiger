/*
 * SliceLayer.h
 *
 *  Created on: Jul 13, 2017
 *      Author: blake
 */

#ifndef INCLUDE_SLICELAYER_H_
#define INCLUDE_SLICELAYER_H_
#include "NeuralLayer.h"
namespace tgr {
enum class SliceType {
	slice_samples, slice_channels
};

/**
 * slice an input data into multiple outputs along a given slice dimension.
 **/
class SliceLayer: public NeuralLayer {
public:
	/**
	 * @param in_shape    [in] size (width * height * channels) of input data
	 * @param slice_type  [in] target axis of slicing
	 * @param num_outputs [in] number of output layers
	 *
	 * example1:
	 *   input:       NxKxWxH = 4x3x2x2  (N:batch-size, K:channels, W:width,
	 *H:height)
	 *   slice_type:  slice_samples
	 *   num_outputs: 3
	 *
	 *   output[0]: 1x3x2x2
	 *   output[1]: 1x3x2x2
	 *   output[2]: 2x3x2x2  (mod data is assigned to the last output)
	 *
	 * example2:
	 *   input:       NxKxWxH = 4x6x2x2
	 *   slice_type:  slice_channels
	 *   num_outputs: 3
	 *
	 *   output[0]: 4x2x2x2
	 *   output[1]: 4x2x2x2
	 *   output[2]: 4x2x2x2
	 **/
	SliceLayer(const aly::dim3& in_shape, SliceType slice_type,int num_outputs);
	SliceLayer(const NeuralLayer& prev_layer, SliceType slice_type,int num_outputs);
	virtual std::vector<aly::dim3> getInputDimensions() const override {
		return {in_shape};
	}
	virtual std::vector<aly::dim3> getOutputDimensions() const override {
		return out_shapes;
	}
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
private:
	void slice_data_forward(const Tensor &in_data,std::vector<Tensor *> &out_data);
	void slice_data_backward(std::vector<Tensor *> &out_grad, Tensor &in_grad);
	void slice_channels_forward(const Tensor &in_data,std::vector<Tensor *> &out_data);
	void slice_channels_backward(std::vector<Tensor *> &out_grad,Tensor &in_grad);
	virtual void setSampleCount(size_t sample_count) override;
	void set_shape();
	void set_shape_data();
	void set_shape_channels();	aly::dim3 in_shape;
	SliceType slice_type;
	int num_outputs;
	std::vector<aly::dim3> out_shapes;
	std::vector<int> slice_size;
};
}

#endif /* INCLUDE_SLICELAYER_H_ */
