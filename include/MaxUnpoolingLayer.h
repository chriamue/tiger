/*
 * MaxUnpoolingLayer.h
 *
 *  Created on: Jul 10, 2017
 *      Author: blake
 */

#ifndef INCLUDE_MAXUNPOOLINGLAYER_H_
#define INCLUDE_MAXUNPOOLINGLAYER_H_
#include "NeuralLayer.h"
namespace tgr {
/**
 * applies max-pooing operaton to the spatial data
 **/
class MaxUnpoolingLayer: public NeuralLayer {
public:
	/**
	 * @param in_width     [in] width of input image
	 * @param in_height    [in] height of input image
	 * @param in_channels  [in] the number of input image channels(depth)
	 * @param unpooling_size [in] factor by which to upscale
	 **/
	MaxUnpoolingLayer(int in_width, int in_height, int in_channels,
			int unpooling_size) :
			MaxUnpoolingLayer(in_width, in_height, in_channels, unpooling_size,
					(in_height == 1 ? 1 : unpooling_size)) {
	}
	MaxUnpoolingLayer(const aly::dim3 &in_size, int unpooling_size, int stride) :
			MaxUnpoolingLayer(in_size.x, in_size.y, in_size.z, unpooling_size,
					(in_size.y == 1 ? 1 : unpooling_size)) {
	}
	MaxUnpoolingLayer(int in_width, int in_height, int in_channels,
			int unpooling_size, int stride);
	virtual int getFanInSize() const override;
	virtual int getFanOutSize() const override;

	virtual void forwardPropagation(
			const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) override;
	virtual void backwardPropagation(
			const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad,
			std::vector<Tensor *> &in_grad)
					override;
	virtual std::vector<aly::dim3> getInputDimensions() const override {
		return {in};
	}
	virtual std::vector<aly::dim3> getOutputDimensions() const override {
		return {out};
	}
	size_t getUnpoolSize() const {
		return unpool_size;
	}
private:
	int unpool_size;
	int stride;
	std::vector<int> out2in;               // mapping out => in (N:1)
	std::vector<std::vector<int>> in2out;  // mapping in => out (1:N)

	struct worker_specific_storage {
		std::vector<int> in2outmax; // mapping max_index(out) => in (1:1)
	};
	worker_specific_storage worker_storage;
	aly::dim3 in;
	aly::dim3 out;
	static int unpool_out_dim(int in_size, int unpooling_size, int stride) {
		return static_cast<int>(static_cast<int64_t>(in_size) * stride
				+ unpooling_size - 1);
	}
	void connect_kernel(int unpooling_size, int inx, int iny, int c);
	void init_connection();
};
}

#endif /* INCLUDE_MAXUNPOOLINGLAYER_H_ */
