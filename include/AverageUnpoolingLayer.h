/*
 * AverageUnpoolingLayer.h
 *
 *  Created on: Jun 26, 2017
 *      Author: blake
 */

#ifndef INCLUDE_AVERAGEUNPOOLINGLAYER_H_
#define INCLUDE_AVERAGEUNPOOLINGLAYER_H_
#include "PartialConnectedLayer.h"

namespace tgr {
/**
 * average pooling with trainable weights
 **/
class AverageUnpoolingLayer: public PartialConnectedLayer {
public:

	/**
	 * @param in_width     [in] width of input image
	 * @param in_height    [in] height of input image
	 * @param in_channels  [in] the number of input image channels(depth)
	 * @param pooling_size [in] factor by which to upscale
	 **/
	AverageUnpoolingLayer(int in_width, int in_height, int in_channels,
			int pooling_size);

	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad)
					override;
private:
	int stride_;
	aly::dim3 in_;
	aly::dim3 out_;
	aly::dim3 w_;
	static int unpool_out_dim(int in_size, int pooling_size, int stride);
	void init_connection(int pooling_size);
	void connect_kernel(int pooling_size, int x, int y, int inc);
};
typedef std::shared_ptr<AverageUnpoolingLayer> AverageUnpoolingLayerPtr;
}

#endif /* INCLUDE_AVERAGEUNPOOLINGLAYER_H_ */
