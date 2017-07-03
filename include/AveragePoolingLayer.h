/*
 * AveragePoolingLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_AVERAGEPOOLINGLAYER_H_
#define INCLUDE_AVERAGEPOOLINGLAYER_H_
#include "PartialConnectedLayer.h"
#include "NeuralSignal.h"
namespace tgr {
class AveragePoolingLayer: public PartialConnectedLayer {
public:
	using Base = PartialConnectedLayer;

	/**
	 * @param in_width     [in] width of input image
	 * @param in_height    [in] height of input image
	 * @param in_channels  [in] the number of input image channels(depth)
	 * @param pool_size_x  [in] factor by which to downscale
	 * @param pool_size_y  [in] factor by which to downscale
	 * @param stride_x     [in] interval at which to apply the filters to the
	 *input
	 * @param stride_y     [in] interval at which to apply the filters to the
	 *input
	 * @param pad_type     [in] padding mode(same/valid)
	 **/
	AveragePoolingLayer(int in_width, int in_height, int in_channels,
			int pool_size_x, int pool_size_y, int stride_x, int stride_y,
			Padding pad_type = Padding::Valid);
	AveragePoolingLayer(int in_width, int in_height, int in_channels,
			int pool_size) :
			AveragePoolingLayer(in_width, in_height, in_channels, pool_size,
					(in_height == 1 ? 1 : pool_size)) {
	}
	AveragePoolingLayer(int in_width, int in_height, int in_channels,
			int pool_size, int stride) :
			AveragePoolingLayer(in_width, in_height, in_channels, pool_size,
					(in_height == 1 ? 1 : pool_size), stride, stride,
					Padding::Valid) {

	}
	virtual void getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual void getStencilWeight(const aly::int3& pos,std::vector<aly::int3>& stencil) const override;
	virtual bool getStencilBias(const aly::int3& pos,aly::int3& stencil) const override;

	virtual std::vector<aly::dim3> getInputDimensions() const override;
	virtual std::vector<aly::dim3> getOutputDimensions() const override;
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)
					override;
private:
	int stride_x_;
	int stride_y_;
	int pool_size_x_;
	int pool_size_y_;
	Padding pad_type_;
	aly::dim3 in_;
	aly::dim3 out_;
	aly::dim3 w_;
	std::pair<int, int> pool_size() const;
	static int pool_out_dim(int in_size, int pooling_size, int stride);
	void init_connection(int pooling_size_x, int pooling_size_y);
	void connect_kernel(int pooling_size_x, int pooling_size_y, int x, int y,
			int inc);
};
typedef std::shared_ptr<AveragePoolingLayer> AveragePoolingLayerPtr;
}

#endif /* INCLUDE_AVERAGEPOOLINGLAYER_H_ */
