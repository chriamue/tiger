#include "NeuralLayer.h"
namespace tgr {
enum class norm_region {
	across_channels, within_channels
};

class LocalResponseNormLayer: public NeuralLayer {
public:
	LocalResponseNormLayer(const aly::dim3 &in_shape, int local_size,
			float alpha = 1.0, float beta = 5.0, norm_region region =
					norm_region::across_channels) :
			NeuralLayer("Local Response Normalization", { ChannelType::data }, {
					ChannelType::data }), in_shape(in_shape), size(local_size), alpha(
					alpha), beta(beta), region(region), in_square(
					in_shape.area()) {
	}
	/**
	 * @param layer       [in] the previous layer connected to this
	 * @param local_size  [in] the number of channels(depths) to sum over
	 * @param in_channels [in] the number of channels of input data
	 * @param alpha       [in] the scaling parameter (same to caffe's LRN)
	 * @param beta        [in] the scaling parameter (same to caffe's LRN)
	 **/
	LocalResponseNormLayer(NeuralLayer *prev, int local_size, float alpha = 1.0,
			float beta = 5.0, norm_region region = norm_region::across_channels) :
			LocalResponseNormLayer(prev->getOutputDimensions()[0], local_size,
					alpha, beta, region) {
	}

	/**
	 * @param in_width    [in] the width of input data
	 * @param in_height   [in] the height of input data
	 * @param local_size  [in] the number of channels(depths) to sum over
	 * @param in_channels [in] the number of channels of input data
	 * @param alpha       [in] the scaling parameter (same to caffe's LRN)
	 * @param beta        [in] the scaling parameter (same to caffe's LRN)
	 **/
	LocalResponseNormLayer(int in_width, int in_height, int local_size,
			int in_channels, float alpha = 1.0, float beta = 5.0,
			norm_region region = norm_region::across_channels) :
			LocalResponseNormLayer(
					aly::dim3 { in_width, in_height, in_channels }, local_size,
					alpha, beta, region) {
	}
	int getFanInSize() const override {
		return size;
	}
	int getFanOutSize() const override {
		return size;
	}
	std::vector<aly::dim3> getInputDimensions() const override {
		return {in_shape};
	}
	std::vector<aly::dim3> getOutputDimensions() const override {
		return {in_shape};
	}
	void forwardPropagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data);
	void backwardPropagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad, std::vector<Tensor *> &in_grad);
private:
	void forward_across(const Storage &in, Storage &out);
	void forward_within(const Storage &in, Storage &out);
	void add_square_sum(const float *src, int size, float *dst);
	void sub_square_sum(const float *src, int size, float *dst);
	aly::dim3 in_shape;
	int size;
	float alpha, beta;
	norm_region region;
	Storage in_square;
};
}
