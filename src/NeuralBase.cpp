/*
 * Copyright(C) 2017, Blake C. Lucas, Ph.D. (img.science@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "NeuralBase.h"
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/parallel_for.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/weight_init.h"
using namespace tiny_dnn;
namespace aly {
Node::Node(int in_size, int out_size) :
		prevEdge(in_size), nextEdge(out_size) {
}
const std::vector<EdgePtr> &Node::previous() const {
	return prevEdge;
}
const std::vector<EdgePtr> &Node::next() const {
	return nextEdge;
}
int Node::previousPort(const Edge &e) const {
	auto it = std::find_if(prevEdge.begin(), prevEdge.end(),
			[&](EdgePtr ep) {return ep.get() == &e;});
	return (int) std::distance(prevEdge.begin(), it);
}
int Node::nextPort(const Edge &e) const {
	auto it = std::find_if(nextEdge.begin(), nextEdge.end(),
			[&](EdgePtr ep) {return ep.get() == &e;});
	return (int) std::distance(nextEdge.begin(), it);
}
std::vector<Node *> Node::previousNodes() const {
	std::vector<Node *> vecs;
	for (auto &e : prevEdge) {
		if (e && e->previous()) {
			vecs.insert(vecs.end(), e->previous());
		}
	}
	return vecs;
}

std::vector<Node *> Node::nextNodes() const {
	std::vector<Node *> vecs;
	for (auto &e : nextEdge) {
		if (e) {
			auto n = e->next();
			vecs.insert(vecs.end(), n.begin(), n.end());
		}
	}
	return vecs;
}

Edge::Edge(Node *prev, const int3 &shape, VectorType vtype) :
		dimensions(shape), vecType(vtype),
		data({Vec1f(ShapeSize(shape))}),
		gradient({Vec1f(ShapeSize(shape))}), previousNode(prev) {
}
void Edge::mergeGradients(Vec1f& dest) {
    Vec1f& grad_head = gradient[0];
    size_t sz  = grad_head.size();
    dest.resize(sz);
    float_t *pdst = &dest[0];
    // dst = grad_[0]
    std::copy(grad_head.data.begin(), grad_head.data.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = gradient.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      vectorize::reduce<float_t>(&gradient[sample][0], sz, pdst);
    }
}
void Edge::clearGradients() {
	for(Vec1f& grad:gradient){
		grad.set(0.0f);
	}
}
Tensor* Edge::getData() {
	return &data;
}
const Tensor *Edge::getData() const {
	return &data;
}
Tensor *Edge::getGradient() {
	return &gradient;
}
const Tensor *Edge::getGradient() const {
	return &gradient;
}
const std::vector<Node *> &Edge::next() const {
	return nextNode;
}
Node *Edge::previous() {
	return previousNode;
}
const Node *Edge::previous() const {
	return previousNode;
}
const int3 &Edge::shape() const {
	return dimensions;
}
VectorType Edge::vtype() const {
	return vecType;
}
void Edge::addNextNode(Node *next) {
	nextNode.push_back(next);
}
Layer::Layer(const std::vector<VectorType> &in_type,
		const std::vector<VectorType> &out_type) :
		Node(static_cast<int>(in_type.size()),
				static_cast<int>(out_type.size())), initialized_(false), parallelize_(
				true), in_channels_(in_type.size()), out_channels_(
				out_type.size()), in_type_(in_type), out_type_(out_type) {
	weight_init_ = std::make_shared<tiny_dnn::weight_init::xavier>();
	bias_init_ = std::make_shared<tiny_dnn::weight_init::constant>();
	trainable_ = true;
	backend_type_=core::backend_t::internal;
}
void Layer::set_parallelize(bool parallelize) {
	parallelize_ = parallelize;
}
void Layer::set_backend(std::shared_ptr<tiny_dnn::core::backend> backend) {
	backend_ = backend;
}
void Layer::set_backend_type(tiny_dnn::core::backend_t backend_type) {
	backend_type_ = backend_type;
}
bool Layer::parallelize() const {
	return parallelize_;
}
tiny_dnn::core::backend_t Layer::backend_type() const {
	return backend_->type();
}

tiny_dnn::core::backend_t Layer::engine() const {
	return backend_type_;
}

std::string Layer::kernel_file() const {
	return std::string("empty_kernel_str");
}
std::string Layer::kernel_header() const {
	return std::string();
}

void Layer::setDevice(const tiny_dnn::Device &device) {
	device_ptr_ = const_cast<tiny_dnn::Device *>(&device);
}

tiny_dnn::Device *Layer::device() const {
	return device_ptr_;
}

std::shared_ptr<tiny_dnn::core::backend> Layer::backend() {
	return backend_;
}

///< number of incoming edges in this Layer
size_t Layer::in_channels() const {
	return in_channels_;
}

///< number of outgoing edges in this Layer
size_t Layer::out_channels() const {
	return out_channels_;
}

int Layer::in_data_size() const {
	return tiny_dnn::sumif(in_shape(), [&](int i) {  // NOLINT
				return in_type_[i] == VectorType::data;
			}, [](const int3 &s) {return s.size();});
}

int Layer::out_data_size() const {
	return tiny_dnn::sumif(out_shape(), [&](int i) {  // NOLINT
				return out_type_[i] == VectorType::data;
			}, [](const int3 &s) {return s.size();});
}

std::vector<int3> Layer::in_data_shape() {
	return tiny_dnn::filter(in_shape(), [&](size_t i) {  // NOLINT
				return in_type_[i] == VectorType::data;
			});
}

std::vector<int3> Layer::out_data_shape() {
	return tiny_dnn::filter(out_shape(), [&](size_t i) {  // NOLINT
				return out_type_[i] == VectorType::data;
			});
}

///! @deprecated use in_data_size() instead
int Layer::in_size() const {
	return in_data_size();
}

///! @deprecated use out_data_size() instead
int Layer::out_size() const {
	return out_data_size();
}

std::vector<const Vec1f *> Layer::weights() const {
	std::vector<const Vec1f *> v;
	for (size_t i = 0; i < in_channels_; i++) {
		if (is_trainable_weight(in_type_[i])) {
			v.push_back(getWeightData(i));
		}
	}
	return v;
}

std::vector<Vec1f *> Layer::weights() {
	std::vector<Vec1f *> v;
	for (size_t i = 0; i < in_channels_; i++) {
		if (is_trainable_weight(in_type_[i])) {
			v.push_back(getWeightData(i));
		}
	}
	return v;
}

std::vector<Tensor *> Layer::weights_grads() {
	std::vector<Tensor *> v;
	for (size_t i = 0; i < in_channels_; i++) {
		if (is_trainable_weight(in_type_[i])) {
			v.push_back(ith_in_node(i)->getGradient());
		}
	}
	return v;
}

std::vector<EdgePtr> Layer::inputs() {
	std::vector<EdgePtr> nodes(in_channels_);
	for (size_t i = 0; i < in_channels_; i++) {
		nodes[i] = ith_in_node(i);
	}
	return nodes;
}

std::vector<EdgePtr> Layer::outputs() {
	std::vector<EdgePtr> nodes(out_channels_);
	for (size_t i = 0; i < out_channels_; i++) {
		nodes[i] = ith_out_node(i);
	}
	return nodes;
}

std::vector<EdgePtr> Layer::outputs() const {
	std::vector<EdgePtr> nodes(out_channels_);
	for (size_t i = 0; i < out_channels_; i++) {
		nodes[i] = const_cast<Layer *>(this)->ith_out_node(i);
	}
	return nodes;
}

void Layer::set_out_grads(const std::vector<const Vec1f *> *grad, size_t cnt) {
		size_t n = 0;
		for (size_t i = 0; i < out_channels_; i++) {
			if (out_type_[i] != VectorType::data)
				continue;
			Tensor &dst_grad = *ith_out_node(i)->getGradient();
			assert(n < cnt);
			const auto &src_grad = grad[n++];
			size_t sz = src_grad.size();
			//dst_grad.resize(src_grad.size());
			for (size_t j = 0; j < sz; ++j) {
				dst_grad[j] = *src_grad[j];
			}
		}
	}

	void Layer::set_in_data(const std::vector<const Vec1f *> *data, size_t cnt) {
		size_t n = 0;
		for (size_t i = 0; i < in_channels_; i++) {
			if (in_type_[i] != VectorType::data)
				continue;
			Tensor &dst_data = *ith_in_node(i)->getData();
			assert(n < cnt);
			const auto &src_data = data[n++];
			size_t sz = src_data.size();
			//dst_data.resize(src_data.soze());
			for (size_t j = 0; j < sz; ++j) {
				dst_data[j] = *src_data[j];
			}
		}
	}

	void Layer::output(std::vector<const Tensor *> &out) const {
		out.clear();
		for (size_t i = 0; i < out_channels_; i++) {
			if (out_type_[i] == VectorType::data) {
				out.push_back(ith_out_node(i)->getData());
			}
		}
	}

	std::vector<VectorType> Layer::in_types() const {
		return in_type_;
	}

	std::vector<VectorType> Layer::out_types() const {
		return out_type_;
	}

	void Layer::set_trainable(bool trainable) {
		trainable_ = trainable;
	}

	bool Layer::trainable() const {
		return trainable_;
	}
	/**
	 * number of incoming connections for each output unit
	 * used only for weight/bias initialization methods which require fan-in
	 *size
	 *(e.g. xavier)
	 * override if the Layer has trainable weights, and scale of initialization
	 *is
	 *important
	 **/
	int Layer::fan_in_size() const {
		return in_shape()[0].x;
	}

	/**
	 * number of outgoing connections for each input unit
	 * used only for weight/bias initialization methods which require fan-out
	 *size
	 *(e.g. xavier)
	 * override if the Layer has trainable weights, and scale of initialization
	 *is
	 *important
	 **/
	int Layer::fan_out_size() const {
		return out_shape()[0].x;
	}


	/* @brief Performs Layer forward operation given an input tensor and
	 * returns the computed data in tensor form.
	 *
	 * @param input Vector of `Tensor` with incoming data.
	 *
	 * Internally, it first allocates data without resetting the weights,
	 * forwards the input data to the computational graph, inside the
	 * forward() method the data from the computational embedded to container
	 * to finally be forwarded to the computational operation kernels.
	 *
	 * Probably there's an overhead of moving from/to the computational
	 * graph. Will be this overhead reduced once we have the Tensor
	 * class integrated?
	 */
	void Layer::forward(const std::vector<Tensor> &input,
			std::vector<const Tensor *> &out) {  // for test
		// allocate data in the computational graph without
		// resetting the weights.
		setup(false);

		std::vector<std::vector<const Vec1f *>> input2;
		input2.resize(input.size());
		for (size_t i = 0; i < input.size(); ++i) {
			input2[i].resize(input[i].size());
			for (size_t j = 0; j < input[i].size(); ++j) {
				input2[i][j] = &input[i][j];
			}
		}

		// the incoming data is forwarded to the computational graph.
		set_in_data(&input2[0], input2.size());
		// pick up the data from the computational graph and perform
		// computation.
		forward();
		// retrieve computed data and return values in form of 4D tensor.
		output(out);
	}

	std::vector<Tensor> Layer::backward(const std::vector<Tensor> &out_grads) { // for test
		setup(false);
		std::vector<std::vector<const Vec1f*>> grads2;
		grads2.resize(out_grads.size());
		for (size_t i = 0; i < out_grads.size(); ++i) {
			grads2[i].resize(out_grads[i].size());
			for (size_t j = 0; j < out_grads[i].size(); ++j) {
				grads2[i][j] = &out_grads[i][j];
			}
		}

		set_out_grads(&grads2[0], grads2.size());
		backward();
		return map_ <Tensor> (inputs(), [](EdgePtr e) {return *e->getGradient();});
	}

	/* @brief The purpose of this method is to forward the data from the
	 * computational graph to the Layer interface.
	 *
	 * This is one of the out of two core (forward/backward) methods that
	 * retrieves the data allocated in the heap by the computational graph
	 * and constructs the containers to handle the computation by batches.
	 * Additionally, the sample count a.k.a number of batches is set.
	 *
	 * Note: in_data and out_data attempt to contain tensors. However, they
	 * are not real tensors since Tensor have three dimensions instead of
	 * four. For this reason they are embedded in to std::vector. Also note
	 * that when std::vector<Tensor*> it's constructed we cannot assure
	 * that data is contiguous.
	 *
	 * After Tensor class integration we should be able to avoid to have
	 * in_data and out_data in vectors since Tensor class itself can handle
	 * batches storage in one single vector with contiguous data.
	 *
	 */
	void Layer::forward() {
		// the computational graph
		fwd_in_data_.resize(in_channels_);
		fwd_out_data_.resize(out_channels_);

		// Organize input/output vectors from storage (computational graph).
		// Internally ith_in_node() will create a connection/edge in the
		// computational graph and will allocate memory in case that it's not
		// done yet.
		for (size_t i = 0; i < in_channels_; i++) {
			fwd_in_data_[i] = ith_in_node(i)->getData();
		}
		// resize outs and stuff to have room for every input sample in
		// the batch
		set_sample_count(fwd_in_data_[0]->size());
		// Internally ith_out_node() will create a connection/edge to the
		// computational graph and will allocate memory in case that it's not
		// done yet. In addition, gradient vector are initialized to default
		// values.
		for (int i = 0; i < out_channels_; i++) {
			fwd_out_data_[i] = ith_out_node(i)->getData();
			ith_out_node(i)->clearGradients();
		}

		// call the forward computation kernel/routine
		forward_propagation(fwd_in_data_, fwd_out_data_);
	}

	void Layer::backward() {
		bwd_in_data_.resize(in_channels_);
		bwd_in_grad_.resize(in_channels_);
		bwd_out_data_.resize(out_channels_);
		bwd_out_grad_.resize(out_channels_);

		// organize input/output vectors from storage
		for (size_t i = 0; i < in_channels_; i++) {
			const auto &nd = ith_in_node(i);
			bwd_in_data_[i] = nd->getData();
			bwd_in_grad_[i] = nd->getGradient();
		}
		for (int i = 0; i < out_channels_; i++) {
			const auto &nd = ith_out_node(i);
			bwd_out_data_[i] = nd->getData();
			bwd_out_grad_[i] = nd->getGradient();
		}
		back_propagation(bwd_in_data_, bwd_out_data_, bwd_out_grad_,
				bwd_in_grad_);
	}

	/* @brief Allocates data in the computational graph and reset weights if
	 * it's needed or the data is not already initialized.
	 *
	 * @param reset_weight Boolean value to force to reset the weights.
	 * Weights will be automatically reset if the are not initialized.
	 *
	 */
	void Layer::setup(bool reset_weight) {
		// The input shape (width x height x depth) must be equal to the number
		// of input channels a.k.a the number of incoming vectors or 'edges' in
		// the computational nomenclature. Same is applied to output shape and
		// numbers of output edges.
		if (in_shape().size() != in_channels_
				|| out_shape().size() != out_channels_) {
			throw nn_error("Connection mismatch at setup Layer");
		}

		// An 'edge' is created in the computational graph from the current
		// Layer/node to each output node and allocates the needed memory.
		// The number of output nodes is determined by the Layer interface.
		// In order to handle graph based networks, which a Layer/node might
		// have multiple input/output connections, we need to check that the
		// connection edge does not already exists if we don't want duplicated
		// memory allocation.
		for (size_t i = 0; i < out_channels_; i++) {
			if (!nextEdge[i]) {
				// connection edge doesn't exist, so we proceed to allocate the
				// necessary memory.
				nextEdge[i] = std::make_shared<Edge>(this, out_shape()[i],out_type_[i]);
			}
		}

		// reset the weights if necessary, or in case that the data is
		// still not initialized.
		if (reset_weight || !initialized_) {
			init_weight();
		}
	}

	/* @brief Initializes the vectors containing the trainable data.
	 *
	 * In case that a Layer/node is set to be not trainable, it does
	 * nothing and returns a void. Otherwise, for each input connection
	 * and depending of the data nature (weight or bias) calls their
	 * pertinent initialization function and fill the vectors with the
	 * data generated by the mentioned functions.
	 *
	 */
	void Layer::init_weight() {
		// Layer/node is not trainable, do nothing and mark the Layer/node
		// as initialized.
		if (!trainable_) {
			initialized_ = true;
			return;
		}

		// Fill vector values with data generated by the initialization
		// function. The pointer to the data is obtained from the
		// computational graph and the methods fan_in_size() and fan_out_size()
		// return the number of incoming/outcoming connections for each
		// input/output unit.
		for (size_t i = 0; i < in_channels_; i++) {
			switch (in_type_[i]) {
			// fill vectors of weight type
			case VectorType::weight:
				weight_init_->fill(&(getWeightData(i)->data), fan_in_size(),fan_out_size());
				break;
				// fill vector of bias type
			case VectorType::bias:
				bias_init_->fill(&(getWeightData(i)->data), fan_in_size(),fan_out_size());
				break;
			default:
				break;
			}
		}
		// in case we succeed with data initialization, we mark the
		// Layer/node as initialized.
		initialized_ = true;
	}

	void Layer::clearGradients() {
		for (int i = 0; i < static_cast<int>(in_type_.size()); i++) {
			ith_in_node(i)->clearGradients();
		}
	}

	void Layer::update_weight(optimizer *o, int batch_size) {
		float_t rcp_batch_size = float_t(1) / float_t(batch_size);
		auto &diff = weights_diff_;
		for (int i = 0; i < static_cast<int>(in_type_.size()); i++) {
			if (trainable() && is_trainable_weight(in_type_[i])) {
				Vec1f &target = *getWeightData(i);
				ith_in_node(i)->mergeGradients(diff);
				for (size_t j = 0; j < diff.size(); ++j) {
					diff[j] *= rcp_batch_size;
				}
				// parallelize only when target size is big enough to mitigate
				// thread spawning overhead.
				bool parallelize = (target.size() >= 512);
				o->update(diff.data, target.data, parallelize);
			}
		}
		clearGradients();
		post_update();
	}

	bool Layer::has_same_weights(const Layer &rhs, float_t eps) const {
		auto w1 = weights();
		auto w2 = rhs.weights();
		if (w1.size() != w2.size())
			return false;

		for (size_t i = 0; i < w1.size(); i++) {
			if (w1[i]->size() != w2[i]->size())
				return false;

			for (size_t j = 0; j < w1[i]->size(); j++) {
				if (std::abs((*w1[i])[j] - (*w2[i])[j]) > eps)
					return false;
			}
		}
		return true;
	}

	void Layer::set_sample_count(size_t sample_count) {
		// increase the size if necessary - but do not decrease
		auto resize = [sample_count](Tensor *tensor) {
			tensor->resize(sample_count, (*tensor)[0]);//
		};
		for (size_t i = 0; i < in_channels_; i++) {
			if (!is_trainable_weight(in_type_[i])) {
				resize(ith_in_node(i)->getData());
			}
			resize(ith_in_node(i)->getGradient());
		}

		for (int i = 0; i < out_channels_; i++) {
			if (!is_trainable_weight(out_type_[i])) {
				resize(ith_out_node(i)->getData());
			}
			resize(ith_out_node(i)->getGradient());
		}
	}
	/* @brief Allocates the necessary edge memory in a specific
		 * incoming connection.
		 *
		 * @param i The position to store the previous edge.
		 *
		 * Graphical explanation:
		 *
		 *     nullptr -- |edge| -- prev(i) ---- |Layer|
		 *               nullptr -- prev(i+1) -´
		 */
		void Layer::alloc_input(int i) const {
			// the created incoming edge won't have a previous connection,
			// for this reason first parameter is a nullptr.
			prevEdge[i] = std::make_shared<Edge>(nullptr, in_shape()[i], in_type_[i]);
		}

		/* @brief Allocates the necessary edge memory in a specific
		 * outcoming connection.
		 *
		 * @param i The position to store the next edge.
		 *
		 * Graphical explanation:
		 *
		 *     |Layer| -- next(i) ---- |edge|
		 *             `- next(i+1) -- nullptr
		 */
		void Layer::alloc_output(int i) const {
			// the created outcoming will have the current Layer as the
			// previous node.
			nextEdge[i] = std::make_shared<Edge>(const_cast<Layer *>(this),
					out_shape()[i], out_type_[i]);
		}

		/* @brief Creates an edge between the current node and one incoming
		 * or previous node.
		 *
		 * @param i The position to store the previous edge.
		 *
		 * The method checks if the edge already exists, otherwise we create it
		 * and the necessary memory it's allocated. The method returns the pointer
		 * to the previous edge.
		 */
		EdgePtr Layer::ith_in_node(int i) {
			// in case that the  edge doesn't exist, we create it
			if (!prevEdge[i])
				alloc_input(i);
			return previous()[i];
		}

		/* @brief Creates an edge between the current node and one outcoming
		 * or next node.
		 *
		 * @param i The position to store the next edge.
		 *
		 * The method checks if the edge already exists, otherwise we create it
		 * and the necessary memory it's allocated. The method returns the pointer
		 * to the next edge.
		 */
		EdgePtr Layer::ith_out_node(int i) {
			// in case that the  edge doesn't exist, we create it
			if (!nextEdge[i])
				alloc_output(i);
			return next()[i];
		}
		EdgePtr Layer::ith_out_node(int i) const {
			return next()[i];
		}

		/* @brief Retrieves weight vector from incoming edge
		 * @param i The position of incoming edge.
		 *
		 * Returns the mutable pointer to the edge raw data.
		 */
		Vec1f *Layer::getWeightData(int i) {
			assert(is_trainable_weight(in_type_[i]));
			return &(*(ith_in_node(i)->getData()))[0];
		}

		/* @brief Retrieves weight vector from incoming edge
		 * @param i The position of incoming edge.
		 *
		 * Returns the non mutable pointer to the edge raw data.
		 */
		const Vec1f *Layer::getWeightData(int i) const {
			assert(is_trainable_weight(in_type_[i]));
			return &(*(const_cast<Layer*>(this)->ith_in_node(i)->getData()))[0];
		}

		void connect(Layer *head, Layer *tail, int head_index,int tail_index) {
			auto out_shape = head->out_shape()[head_index];
			auto in_shape = tail->in_shape()[tail_index];
			head->setup(false);
			// (karandesai) enable shape inferring for all layers
			// currently only possible for activation layers.
			if (in_shape.size() == 0) {
				tail->set_in_shape(out_shape);
				in_shape = out_shape;
			}

			if (out_shape.size() != in_shape.size()) {
				connection_mismatch(*head, *tail);
			}

			if (!head->nextEdge[head_index]) {
				throw nn_error("output edge must not be null");
			}
			tail->prevEdge[tail_index] = head->nextEdge[head_index];
			tail->prevEdge[tail_index]->addNextNode(tail);
		}
}
