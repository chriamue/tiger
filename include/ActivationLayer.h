/*
 * ActivationLayer.h
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#ifndef INCLUDE_ACTIVATIONLAYER_H_
#define INCLUDE_ACTIVATIONLAYER_H_

#include "NeuralLayer.h"
#include <AlloyMath.h>
namespace tgr {
class ActivationLayer: public NeuralLayer {
public:
	/**
	 * Construct an activation layer which will take shape when connected to some
	 * layer. Connection happens like ( layer1 << act_layer1 ) and shape of this
	 * layer is inferred at that time.
	 */
	ActivationLayer(const std::string& name) :
			ActivationLayer(name, aly::dim3(0, 0, 0)) {
	}

	/**
	 * Construct a flat activation layer with specified number of neurons.
	 * This constructor is suitable for adding an activation layer after
	 * flat layers such as fully connected layers.
	 *
	 * @param in_dim      [in] number of elements of the input
	 */
	ActivationLayer(const std::string& name, int in_dim) :
			ActivationLayer(name, aly::dim3(in_dim, 1, 1)) {
	}

	/**
	 * Construct an activation layer with specified width, height and channels.
	 * This constructor is suitable for adding an activation layer after spatial
	 * layers such as convolution / pooling layers.
	 *
	 * @param in_width    [in] number of input elements along width
	 * @param in_height   [in] number of input elements along height
	 * @param in_channels [in] number of channels (input elements along depth)
	 */
	ActivationLayer(const std::string& name, int in_width, int in_height,
			int in_channels) :
			ActivationLayer(name, aly::dim3(in_width, in_height, in_channels)) {
	}

	/**
	 * Construct an activation layer with specified input shape.
	 *
	 * @param in_shape [in] shape of input tensor
	 */
	ActivationLayer(const std::string& name, const aly::dim3 &in_shape) :
			NeuralLayer(name, { ChannelType::data }, { ChannelType::data }), in_shape(
					in_shape) {
	}

	/**
	 * Construct an activation layer given the previous layer.
	 * @param prev_layer previous layer
	 */
	ActivationLayer(const std::string& name, const NeuralLayer &prev_layer) :
			NeuralLayer(name, { ChannelType::data }, { ChannelType::data }), in_shape(
					prev_layer.getOutputDimensions(0)) {
	}

	virtual std::vector<aly::dim3> getInputDimensions() const override {
		return {in_shape};
	}

	virtual std::vector<aly::dim3> getOutputDimensions() const override {
		return {in_shape};
	}
	virtual void getStencilInput(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override {
		stencil = std::vector<aly::int3> { pos };
	}
	virtual void getStencilWeight(const aly::int3& pos,
			std::vector<aly::int3>& stencil) const override {
		stencil.clear();
	}
	virtual bool getStencilBias(const aly::int3& pos, aly::int3& stencil) const
			override {
		return false;
	}
	virtual void setInputShape(const aly::dim3& in_shape) override {
		this->in_shape = in_shape;
	}
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;

	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)
					override;

	/**
	 * Populate Storage of elements 'y' according to activation y = f(x).
	 * Child classes must override this method, apply activation function
	 * element wise over a Storage of elements.
	 *
	 * @param x  input vector
	 * @param y  output vector (values to be assigned based on input)
	 **/
	virtual void forward_activation(const Storage &x, Storage &y) = 0;

	/**
	 * Populate Storage of elements 'dx' according to gradient of activation.
	 *
	 * @param x  input vector of current layer (same as forward_activation)
	 * @param y  output vector of current layer (same as forward_activation)
	 * @param dx gradient of input vectors (i-th element correspond with x[i])
	 * @param dy gradient of output vectors (i-th element correspond with y[i])
	 **/
	virtual void backward_activation(const Storage &x, const Storage &y,
			Storage &dx, const Storage &dy) = 0;
	/**
	 * Target value range for learning.
	 */
	virtual std::pair<float_t, float_t> scale() const = 0;

private:
	aly::dim3 in_shape;
};

typedef std::shared_ptr<ActivationLayer> ActivationLayerPtr;
}

#endif /* INCLUDE_ACTIVATIONLAYER_H_ */
