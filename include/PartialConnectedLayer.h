/*
 * PartialConnectedLayer.h
 *
 *  Created on: Jun 25, 2017
 *      Author: blake
 */

#ifndef INCLUDE_PARTIALCONNECTEDLAYER_H_
#define INCLUDE_PARTIALCONNECTEDLAYER_H_

#include "NeuralLayer.h"
#include "NeuralSignal.h"
namespace tgr {
class PartialConnectedLayer: public NeuralLayer {
public:
	typedef std::vector<std::pair<int, int>> io_connections;
	typedef std::vector<std::pair<int, int>> wi_connections;
	typedef std::vector<std::pair<int, int>> wo_connections;
	PartialConnectedLayer(const std::string& name,int in_dim, int out_dim, size_t weight_dim,
			size_t bias_dim, float scale_factor = 1.0f);
	virtual void forwardPropagation(const std::vector<Tensor*>&in_data,
			std::vector<Tensor*> &out_data) override;
	virtual void backwardPropagation(const std::vector<Tensor*> &in_data,
			const std::vector<Tensor*> &out_data,
			std::vector<Tensor*> &out_grad, std::vector<Tensor*> &in_grad)
					override;
	virtual int getFanInSize() const override;
	virtual int getFanOutSize() const override;
	virtual void getStencilInput(const aly::int3& pos,std::vector<aly::int3>& stencil) const =0;
	virtual void getStencilWeight(const aly::int3& pos,std::vector<aly::int3>& stencil) const =0;
	virtual bool getStencilBias(const aly::int3& pos,aly::int3& stencil) const =0;

private:
	size_t param_size() const;
protected:
	void connect_weight(int input_index, int output_index, int weight_index);
	void connect_bias(int bias_index, int output_index);
	std::vector<io_connections> weight2io_;  // weight_id -> [(in_id, out_id)]
	std::vector<wi_connections> out2wi_;     // out_id -> [(weight_id, in_id)]
	std::vector<wo_connections> in2wo_;      // in_id -> [(weight_id, out_id)]
	std::vector<std::vector<int>> bias2out_;
	std::vector<size_t> out2bias_;
	float_t scale_factor_;
};
typedef std::shared_ptr<PartialConnectedLayer> PartialConnectedLayerPtr;
}

#endif /* INCLUDE_PARTIALCONNECTEDLAYER_H_ */
