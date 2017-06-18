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
#ifndef NEURAL_BASE_H_
#define NEURAL_BASE_H_

#include <AlloyOptimizationMath.h>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/framework/device.fwd.h"


namespace aly {

enum class VectorType
	: int32_t {
		// 0x0001XXX : in/out data
	data = 0x0001000,  // input/output data, fed by other Layer or input channel

	// 0x0002XXX : trainable parameters, updated for each back propagation
	weight = 0x0002000,
	bias = 0x0002001,

	label = 0x0004000,
	aux = 0x0010000  // Layer-specific storage
};
inline bool is_trainable_weight(enum VectorType vtype) {
	return (static_cast<int>(vtype) & static_cast<int>(VectorType::weight)) == static_cast<int>(VectorType::weight);
}
typedef std::vector<Vec1f> Tensor;
class Node;
class Layer;
class Edge;
typedef std::shared_ptr<Edge> EdgePtr;
void connect(Layer *head, Layer *tail, int head_index = 0, int tail_index = 0);
/**
 * base class of all kind of tinny-cnn data
 **/
class Node: public std::enable_shared_from_this<Node> {
public:
	Node(int in_size, int out_size);
	virtual ~Node() {
	}
	const std::vector<EdgePtr> &previous() const;
	const std::vector<EdgePtr> &next() const;
	int previousPort(const Edge &e) const;
	int nextPort(const Edge &e) const;
	std::vector<Node *> previousNodes() const; // @todo refactor and remove this method
	std::vector<Node *> nextNodes() const; // @todo refactor and remove this method
protected:
	Node() = delete;
	friend void connect(Layer *head, Layer *tail, int head_index, int tail_index);
	mutable std::vector<EdgePtr> prevEdge;
	mutable std::vector<EdgePtr> nextEdge;
};

/**
 * class containing input/output data
 **/

inline size_t ShapeSize(int3 dims){
	return (size_t)dims.x*(size_t)dims.y*(size_t)dims.z;
}
class Edge {
public:
	Edge(Node *prev, const int3 &shape, VectorType vtype);
	void mergeGradients(Vec1f& dest);
	void clearGradients();
	Tensor* getData();
	const Tensor* getData() const;
	Tensor *getGradient();
	const Tensor *getGradient() const;
	const std::vector<Node *> &next() const;
	Node *previous();
	const Node *previous() const;
	const int3 &shape() const;
	VectorType vtype() const;
	void addNextNode(Node *next);

private:
	int3 dimensions;
	VectorType vecType;
	Tensor data;
	Tensor gradient;
	Node *previousNode;  // previous Node, "producer" of this tensor
	std::vector<Node *> nextNode;  // next nodes, "consumers" of this tensor
};

template<typename T> struct layer_tuple {
	layer_tuple(T l1, T l2) {
		layers.push_back(l1);
		layers.push_back(l2);
	}
	std::vector<T> layers;
};

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of Layer
 **/
class Layer: public Node {
public:
	friend void connection_mismatch(const Layer &from, const Layer &to);

	virtual ~Layer() = default;

	/**
	 * @brief Defaul Layer constructor that instantiates a N-input, M-output
	 *Layer
	 *
	 * @param in_type[N] type of input vector (data, weight, bias...)
	 * @param out_type[M] type of output vector
	 *
	 **/
	Layer(const std::vector<VectorType> &in_type,
			const std::vector<VectorType> &out_type);
	Layer(const Layer &) = default;
	Layer &operator=(const Layer &) = default;
	Layer(Layer &&) = default;
	Layer &operator=(Layer &&) = default;
	void set_parallelize(bool parallelize);
	void set_backend(std::shared_ptr<tiny_dnn::core::backend> backend);
	void set_backend_type(tiny_dnn::core::backend_t backend_type);
	/////////////////////////////////////////////////////////////////////////
	// getter
	bool parallelize() const;
	// TODO(edgar): Deprecated: use the below method
	tiny_dnn::core::backend_t backend_type() const;
	tiny_dnn::core::backend_t engine() const;
	virtual std::string kernel_file() const;
	virtual std::string kernel_header() const;
	virtual void createOp() {
	}
	void setDevice(const tiny_dnn::Device &device);

	tiny_dnn::Device *device() const;
	std::shared_ptr<tiny_dnn::core::backend> backend();
	///< number of incoming edges in this Layer
	size_t in_channels() const;
	///< number of outgoing edges in this Layer
	size_t out_channels() const;
	int in_data_size() const;
	int out_data_size() const;
	std::vector<int3> in_data_shape();

	std::vector<int3> out_data_shape();

	///! @deprecated use in_data_size() instead
	int in_size() const;
	///! @deprecated use out_data_size() instead
	int out_size() const;
	std::vector<const Vec1f *> weights() const;
	std::vector<Vec1f *> weights();
	std::vector<Tensor *> weights_grads();
	std::vector<EdgePtr> inputs() ;
	std::vector<EdgePtr> outputs() ;
	std::vector<EdgePtr> outputs() const;
	void set_out_grads(const std::vector<const Vec1f *> *grad, size_t cnt);
	void set_in_data(const std::vector<const Vec1f *> *data, size_t cnt);
	void output(std::vector<const Tensor *> &out) const ;
	std::vector<VectorType> in_types() const ;
	std::vector<VectorType> out_types() const ;
	void set_trainable(bool trainable);
	bool trainable() const;

	/**
	 * return output value range
	 * used only for calculating target value from label-id in final(output)
	 *Layer
	 * override properly if the Layer is intended to be used as output Layer
	 **/
	virtual std::pair<float_t, float_t> out_value_range() const {
		return {float_t {0.0}, float_t {1.0}};
	}

	/**
	 * array of input shapes (width x height x depth)
	 **/
	virtual std::vector<int3> in_shape() const = 0;

	/**
	 * set input shape of a Layer (only used internally while shape inferring)
	 */
	virtual void set_in_shape(const int3 &in_shape) {
		throw std::runtime_error(
				"Can't set shape. Shape inferring not applicable for this "
						"Layer (yet).");
	}
	;

	/**
	 * array of output shapes (width x height x depth)
	 **/
	virtual std::vector<int3> out_shape() const = 0;

	/**
	 * name of Layer, should be unique for each concrete class
	 **/
	virtual std::string layer_type() const = 0;

	/**
	 * number of incoming connections for each output unit
	 * used only for weight/bias initialization methods which require fan-in
	 *size
	 *(e.g. xavier)
	 * override if the Layer has trainable weights, and scale of initialization
	 *is
	 *important
	 **/
	virtual int fan_in_size() const;

	/**
	 * number of outgoing connections for each input unit
	 * used only for weight/bias initialization methods which require fan-out
	 *size
	 *(e.g. xavier)
	 * override if the Layer has trainable weights, and scale of initialization
	 *is
	 *important
	 **/
	virtual int fan_out_size() const;

	/////////////////////////////////////////////////////////////////////////
	// setter
	template<typename WeightInit>
	Layer &weight_init(const WeightInit &f) {
		weight_init_ = std::make_shared<WeightInit>(f);
		return *this;
	}

	template<typename BiasInit>
	Layer &bias_init(const BiasInit &f) {
		bias_init_ = std::make_shared<BiasInit>(f);
		return *this;
	}

	template<typename WeightInit>
	Layer &weight_init(std::shared_ptr<WeightInit> f) {
		weight_init_ = f;
		return *this;
	}

	template<typename BiasInit>
	Layer &bias_init(std::shared_ptr<BiasInit> f) {
		bias_init_ = f;
		return *this;
	}

	/////////////////////////////////////////////////////////////////////////
	// save/load
	template<typename Archive>
	void serialize(Archive &ar) {
		auto all_weights = weights();
		for (auto weight : all_weights) {
			ar(*weight);
		}
		initialized_ = true;
	}

	virtual void save(std::ostream &os,
			const int precision = std::numeric_limits<float_t>::digits10 + 2
			/*by default, we want there to be enough precision*/) const { // NOLINT

			/*
			 if (is_exploded()) {
			 throw std::runtime_error("failed to save weights because of infinite weight");
			 }*/
		os << std::setprecision(precision);
		auto all_weights = weights();
		for (auto &weight : all_weights) {
			for (auto w : weight->data)
				os << w << " ";
		}
	}

	virtual void load(std::istream &is,
			const int precision = std::numeric_limits<float_t>::digits10 + 2
			/*by default, we want there to be enough precision*/) {  // NOLINT
		is >> std::setprecision(precision);
		auto all_weights = weights();
		for (auto &weight : all_weights) {
			for (auto &w : weight->data)
				is >> w;
		}
		initialized_ = true;
	}

	virtual void load(const std::vector<float_t> &src, int &idx) {  // NOLINT
		auto all_weights = weights();
		for (auto &weight : all_weights) {
			for (auto &w : weight->data)
				w = src[idx++];
		}
		initialized_ = true;
	}

/////////////////////////////////////////////////////////////////////////
// visualize

///< visualize latest output of this Layer
///< default implementation interpret output as 1d-vector,
///< so "visual" Layer(like convolutional Layer) should override this for better
/// visualization.
#ifdef DNN_USE_IMAGE_API
	virtual image<> output_to_image(size_t channel = 0) const {
		const Vec1f *output = &(*(outputs()[channel]->get_data()))[0];
		return vec2image<unsigned char>(*output, out_shape()[channel]);
	}
#endif

	/////////////////////////////////////////////////////////////////////////
	// fprop/bprop

	/**
	 * @param in_data  input vectors of this Layer (data, weight, bias)
	 * @param out_data output vectors
	 **/
	virtual void forward_propagation(const std::vector<Tensor *> &in_data,
			std::vector<Tensor *> &out_data) = 0;

	/**
	 * return delta of previous Layer (delta=\frac{dE}{da}, a=wx in
	 *fully-connected Layer)
	 * @param in_data  input vectors (same vectors as forward_propagation)
	 * @param out_data output vectors (same vectors as forward_propagation)
	 * @param out_grad gradient of output vectors (i-th vector correspond with
	 *out_data[i])
	 * @param in_grad  gradient of input vectors (i-th vector correspond with
	 *in_data[i])
	 **/
	virtual void back_propagation(const std::vector<Tensor *> &in_data,
			const std::vector<Tensor *> &out_data,
			std::vector<Tensor *> &out_grad,
			std::vector<Tensor *> &in_grad) = 0;

	/**
	 * return delta2 of previous Layer (delta2=\frac{d^2E}{da^2}, diagonal of
	 *hessian matrix)
	 * it is never called if optimizer is hessian-free
	 **/
	// virtual void back_propagation_2nd(const std::vector<Vec1f>& delta_in) =
	// 0;
	// called afrer updating weight
	virtual void post_update() {
	}

	/**
	 * notify changing context (train <=> test)
	 **/
	virtual void set_context(tiny_dnn::net_phase ctx) {
		CNN_UNREFERENCED_PARAMETER(ctx);
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
	 * TODO: Probably there's an overhead of moving from/to the computational
	 * graph. Will be this overhead reduced once we have the Tensor
	 * class integrated?
	 */
	void forward(const std::vector<Tensor> &input,std::vector<const Tensor *> &out) ;
	std::vector<Tensor> backward(const std::vector<Tensor> &out_grads) ;

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
	void forward();

	void backward();
	/* @brief Allocates data in the computational graph and reset weights if
	 * it's needed or the data is not already initialized.
	 *
	 * @param reset_weight Boolean value to force to reset the weights.
	 * Weights will be automatically reset if the are not initialized.
	 *
	 */
	void setup(bool reset_weight) ;

	/* @brief Initializes the vectors containing the trainable data.
	 *
	 * In case that a Layer/node is set to be not trainable, it does
	 * nothing and returns a void. Otherwise, for each input connection
	 * and depending of the data nature (weight or bias) calls their
	 * pertinent initialization function and fill the vectors with the
	 * data generated by the mentioned functions.
	 *
	 */
	void init_weight();
	void clearGradients();
	void update_weight(tiny_dnn::optimizer *o, int batch_size);
	bool has_same_weights(const Layer &rhs, float_t eps) const;
	virtual void set_sample_count(size_t sample_count);

	/**
	 * generate Layer from cereal's Archive
	 **/
	template<typename InputArchive>
	static std::shared_ptr<Layer> load_layer(InputArchive &ia);

	template<typename OutputArchive>
	static void save_layer(OutputArchive &oa, const Layer &l);

	template<class Archive>
	void serialize_prolog(Archive &ar);

protected:
	/** Flag indication whether the Layer/node is initialized */
	bool initialized_;
	/** Flag indicating whether the Layer/node operations ara paralellized */
	bool parallelize_;
	/** The number of input vectors/edges */
	size_t in_channels_;
	/** The number of output vectors/edges */
	size_t out_channels_;
	/** Vector containing the type of data for inputs */
	std::vector<VectorType> in_type_;
	/** Vector containing the type of data for outputs */
	std::vector<VectorType> out_type_;
	/** The current backend type for operations */
	tiny_dnn::core::backend_t backend_type_;
	/** The backend instance (deprecated) */
	std::shared_ptr<tiny_dnn::core::backend> backend_;
	/** Pointer to the device on which the Layer/node will run */
	tiny_dnn::Device *device_ptr_ = nullptr;
	/** Used in update_weight method. Kept as a member variable to reduce
	 * frequent
	 * memory allocation */
	Vec1f weights_diff_;

	template<typename T, typename Func>
	inline void for_i(T size, Func f, size_t grainsize = 100) {
		tiny_dnn::for_i(parallelize_, size, f, grainsize);
	}

private:
	/** Flag indicating whether the Layer/node parameters are trainable */
	bool trainable_;
	/** Pointer to the function for weights initialization */
	std::shared_ptr<tiny_dnn::weight_init::function> weight_init_;
	/** Pointer to the function for biases initialization */
	std::shared_ptr<tiny_dnn::weight_init::function> bias_init_;

	std::vector<Tensor *> fwd_in_data_;
	std::vector<Tensor *> fwd_out_data_;
	std::vector<Tensor *> bwd_in_data_;
	std::vector<Tensor *> bwd_in_grad_;
	std::vector<Tensor *> bwd_out_data_;
	std::vector<Tensor *> bwd_out_grad_;

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
	void alloc_input(int i) const;

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
	void alloc_output(int i) const;

	/* @brief Creates an edge between the current node and one incoming
	 * or previous node.
	 *
	 * @param i The position to store the previous edge.
	 *
	 * The method checks if the edge already exists, otherwise we create it
	 * and the necessary memory it's allocated. The method returns the pointer
	 * to the previous edge.
	 */
	EdgePtr ith_in_node(int i);

	/* @brief Creates an edge between the current node and one outcoming
	 * or next node.
	 *
	 * @param i The position to store the next edge.
	 *
	 * The method checks if the edge already exists, otherwise we create it
	 * and the necessary memory it's allocated. The method returns the pointer
	 * to the next edge.
	 */
	EdgePtr ith_out_node(int i);
	EdgePtr ith_out_node(int i) const;

	/* @brief Retrieves weight vector from incoming edge
	 * @param i The position of incoming edge.
	 *
	 * Returns the mutable pointer to the edge raw data.
	 */
	Vec1f *getWeightData(int i);

	/* @brief Retrieves weight vector from incoming edge
	 * @param i The position of incoming edge.
	 *
	 * Returns the non mutable pointer to the edge raw data.
	 */
	const Vec1f *getWeightData(int i) const;
};

inline Layer &operator<<(Layer &lhs, Layer &rhs) {
	connect(&lhs, &rhs,0,0);
	return rhs;
}
template<class C, class R> std::basic_ostream<C, R> & operator <<(
	std::basic_ostream<C, R> & ss, const std::vector<int3>& dims) {
	for(int n=0,N=dims.size();n<N;n++){
		if(n<=N-1){
			ss<<dims[n]<<",";
		} else {
			ss<<dims[n];
		}
	}
	return ss;
}
template<typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits> &operator<<(
		std::basic_ostream<Char, CharTraits> &os, const Layer &v) {
	v.save(os);
	return os;
}

template<typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits> &operator>>(
		std::basic_istream<Char, CharTraits> &os, Layer &v) {
	v.load(os);
	return os;
}

// error message functions

inline void connection_mismatch(const Layer &from, const Layer &to) {
	std::ostringstream os;

	os << std::endl;
	os << "output size of Nth Layer must be equal to input of (N+1)th Layer\n";

	os << "layerN:   " << std::setw(12) << from.layer_type() << " in:"
			<< from.in_data_size() << "(" << from.in_shape() << "), " << "out:"
			<< from.out_data_size() << "(" << from.out_shape() << ")\n";

	os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:"
			<< to.in_data_size() << "(" << to.in_shape() << "), " << "out:"
			<< to.out_data_size() << "(" << to.out_shape() << ")\n";

	os << from.out_data_size() << " != " << to.in_data_size() << std::endl;
	std::string detail_info = os.str();

	throw std::runtime_error("Layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const Layer &Layer, const Vec1f &data) {
	std::ostringstream os;

	os << std::endl;
	os << "data dimension:    " << data.size() << "\n";
	os << "network dimension: " << Layer.in_data_size() << "("
			<< Layer.layer_type() << ":" << Layer.in_shape() << ")\n";

	std::string detail_info = os.str();

	throw std::runtime_error("input dimension mismatch!" + detail_info);
}

inline void pooling_size_mismatch(int in_width, int in_height,
		int pooling_size_x, int pooling_size_y) {
	std::ostringstream os;

	os << std::endl;
	os << "WxH:" << in_width << "x" << in_height << std::endl;
	os << "pooling-size:" << pooling_size_x << "x" << pooling_size_y
			<< std::endl;

	std::string detail_info = os.str();

	throw std::runtime_error("width/height not multiple of pooling size" + detail_info);
}

template<typename T, typename U>
void graph_traverse(Layer *root_node, T &&node_callback, U &&edge_callback) {
	std::unordered_set<Layer *> visited;
	std::queue<Layer *> S;

	S.push(root_node);

	while (!S.empty()) {
		Layer *curr = S.front();
		S.pop();
		visited.insert(curr);

		node_callback(*curr);

		auto edges = curr->next();
		for (auto e : edges) {
			if (e != nullptr)
				edge_callback(*e);
		}

		auto prev = curr->previousNodes();
		for (auto p : prev) {
			// TODO(nyanp): refactoring
			// which type of refactoring do you have in mind for that?
			Layer *l = dynamic_cast<Layer *>(p);
			if (visited.find(l) == visited.end()) {
				S.push(l);
			}
		}

		auto next = curr->nextNodes();
		for (auto n : next) {
			// TODO(nyanp): refactoring
			// which type of refactoring do you have in mind for that?
			Layer *l = dynamic_cast<Layer *>(n);
			if (visited.find(l) == visited.end()) {
				S.push(l);
			}
		}
	}
}

template<typename T, typename U,
		typename std::enable_if<
				std::is_base_of<Layer, T>::value
						&& std::is_base_of<Layer, U>::value>::type * = nullptr>
layer_tuple<Layer *> operator,(T &l1, U &l2) {
	return layer_tuple<Layer *>(&l1, &l2);
}

template<typename T, typename U,
		typename std::enable_if<
				std::is_base_of<Layer, T>::value
						&& std::is_base_of<Layer, U>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<Layer>> operator,(std::shared_ptr<T> l1,
		std::shared_ptr<U> l2) {
	return layer_tuple<std::shared_ptr<Layer>>(l1, l2);
}

template<typename T,
		typename std::enable_if<std::is_base_of<Layer, T>::value>::type * =
				nullptr>
layer_tuple<Layer *> operator,(layer_tuple<Layer *> lhs, T &rhs) {
	lhs.layers.push_back(&rhs);
	return lhs;
}

template<typename T,
		typename std::enable_if<std::is_base_of<Layer, T>::value>::type * =
				nullptr>
layer_tuple<std::shared_ptr<Layer>> operator,(
		layer_tuple<std::shared_ptr<Layer>> lhs, std::shared_ptr<T> &rhs) {
	lhs.layers.push_back(rhs);
	return lhs;
}

template<typename T,
		typename std::enable_if<std::is_base_of<Layer, T>::value>::type * =
				nullptr>
layer_tuple<Layer *> operator,(T &lhs, layer_tuple<Layer *> rhs) {
	rhs.layers.insert(rhs.layers.begin(), &lhs);
	return rhs;
}

template<typename T,
		typename std::enable_if<std::is_base_of<Layer, T>::value>::type * =
				nullptr>
layer_tuple<std::shared_ptr<Layer>> operator,(std::shared_ptr<T> &lhs,
		layer_tuple<std::shared_ptr<Layer>> rhs) {
	rhs.layers.insert(rhs.layers.begin(), lhs);
	return rhs;
}

template<typename T, typename U>
inline std::shared_ptr<U> &operator<<(std::shared_ptr<T> &lhs,
		std::shared_ptr<U> &rhs) {
	connect(lhs.get(), rhs.get());
	return rhs;
}

template<typename T>
inline T &operator<<(const layer_tuple<std::shared_ptr<Layer>> &lhs, T &rhs) {
	for (size_t i = 0; i < lhs.layers.size(); i++) {
		connect(&*lhs.layers[i], &*rhs, 0, i);
	}
	return rhs;
}

template<typename T>
inline const layer_tuple<std::shared_ptr<Layer>> &operator<<(T &lhs,
		const layer_tuple<std::shared_ptr<Layer>> &rhs) {
	for (size_t i = 0; i < rhs.layers.size(); i++) {
		connect(&*lhs, &*rhs.layers[i], i, 0);
	}
	return rhs;
}

template<typename T>
inline T &operator<<(const layer_tuple<Layer *> &lhs, T &rhs) {
	for (size_t i = 0; i < lhs.layers.size(); i++) {
		connect(lhs.layers[i], &rhs, 0, i);
	}
	return rhs;
}

template<typename T>
inline const layer_tuple<Layer *> &operator<<(T &lhs,
		const layer_tuple<Layer *> &rhs) {
	for (size_t i = 0; i < rhs.layers.size(); i++) {
		connect(&lhs, rhs.layers[i], i, 0);
	}
	return rhs;
}
}  // namespace tiny_dnn
#endif
