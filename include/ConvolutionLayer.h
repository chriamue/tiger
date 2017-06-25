/*
 * ConvolutionLayer.h
 *
 *  Created on: Jun 24, 2017
 *      Author: blake
 */

#ifndef INCLUDE_CONVOLUTIONLAYER_H_
#define INCLUDE_CONVOLUTIONLAYER_H_

#include "NeuralLayer.h"
#include "NeuralSignal.h"

#include "tiny_dnn/tiny_dnn.h"
#include <AlloyImage.h>
#include <AlloyMath.h>

namespace tgr {
enum class Padding {
	Valid = tiny_dnn::padding::valid, Same = tiny_dnn::padding::same
};
class ConvolutionLayer: public NeuralLayer {
	tiny_dnn::convolutional_layer dnnLayer;
	ConvolutionLayer(int in_width, int in_height, int window_width,int window_height, int in_channels, int out_channels,const tiny_dnn::core::connection_table& connection_table,Padding pad_type = Padding::Valid, bool has_bias = true,int w_stride = 1, int h_stride = 1, BackendType backend_type = DefaultEngine());
	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void setInputShape(const aly::int3& in_shape) override;
	virtual void post() override;
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,std::vector<Tensor*> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,const std::vector<Tensor*> &out_data,std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)override;
	virtual void setSampleCount(size_t sample_count) override;
	virtual int getFanInSize() const override;
	virtual int getFanOutSize() const override;
};

}

#endif /* INCLUDE_CONVOLUTIONLAYER_H_ */
