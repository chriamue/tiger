#include "tiny_dnn/node.h"
#include "tiny_dnn/layers/layer.h"
#include <AlloyMath.h>
namespace aly {
class Layer {
protected:
	tiny_dnn::layer* layer;
	bool visited;
	int id;
	std::vector<Layer*> children;
	std::vector<Layer*> dependencies;
public:
	size_t getInputSize() const {
		return layer->in_shape().size();
	}
	int3 getInputShapeSize(size_t idx) const {
		tiny_dnn::shape3d shape = layer->in_shape()[idx];
		return int3(shape.width_, shape.height_, shape.depth_);
	}
	const std::vector<tiny_dnn::shape3d>& getOutputShape() const {
		return layer->out_shape();
	}
	size_t getOutputSize() const {
		return layer->in_shape().size();
	}
	int3 getOutputShapeSize(size_t idx) const {
		tiny_dnn::shape3d shape = layer->in_shape()[idx];
		return int3(shape.width_, shape.height_, shape.depth_);
	}
	std::vector<tiny_dnn::shape3d> getInputShape() {
		return layer->in_shape();
	}
	std::vector<tiny_dnn::shape3d> getOutputShape() {
		return layer->out_shape();
	}
	int getInputChannels() const {
		return layer->in_channels();
	}
	int getOutputChannels() const {
		return layer->out_channels();
	}
	tiny_dnn::tensor_t& getInputWeights(int i) {
		return *layer->ith_in_node(i)->get_data();
	}
	tiny_dnn::tensor_t& getOutputWeights(int i) {
		return *layer->ith_out_node(i)->get_data();
	}
	bool isTrainable() const {
		return layer->trainable();
	}
	bool isVisited() const {
		return visited;
	}
	void setVisited(bool b) {
		visited = b;
	}
	const std::vector<std::shared_ptr<Layer>>& getChildren() const {
		return children;
	}
	const std::vector<Layer*>& getDependencies() const {
		return dependencies;
	}
	bool isRoot() const {
		return (dependencies.size() == 0);
	}
	bool isLeaf() const {
		return (children.size() == 0);
	}
	void addChild(Layer* layer) {
		children.push_back(layer);
		layer->dependencies.push_back(this);
	}
	void setId(int id) {
		this->id = id;
	}
	int getId() const {
		return id;
	}
	Layer(tiny_dnn::layer* layer) :
			layer(layer), visited(false), id(-1) {
	}
};
typedef std::shared_ptr<Layer> LayerPtr;
}
