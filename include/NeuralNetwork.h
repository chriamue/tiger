/*
 * NeuralNetwork.h
 *
 *  Created on: Jun 18, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURALNETWORK_H_
#define INCLUDE_NEURALNETWORK_H_
#include <tiny_dnn/tiny_dnn.h>
#include "NeuralBase.h"
namespace aly {
class NeuralNetwork {
protected:
	std::vector<std::shared_ptr<Layer>> layers;
	std::vector<std::shared_ptr<Layer>> roots;
	std::vector<std::shared_ptr<Layer>> leafs;
public:
	void build(tiny_dnn::network<tiny_dnn::graph>& net) {
		int N=net.layer_size();
		std::map<tiny_dnn::layer*,int> layerMap;
		for(int n=0;n<N;n++){
			layerMap[net[n]]=n;
			LayerPtr l=aly::MakeShared<Layer>(net[n]);
			l->setId(n);
			layers.push_back(l);
		}
		std::list<LayerPtr> q;
		for(tiny_dnn::layer* layer:net.net_.input_layers()){
			LayerPtr l=layers[layerMap[layer]];
			roots.push_back(l);
			std::cout<<"Found Root "<<l->getId()<<std::endl;
			for(tiny_dnn::node* child:layer->next_nodes()){

			}
		}
		/*
		for (LayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isRoot()) {
				roots.push_back(layer);
				q.push_back(layer);
			}
		}
		int index = 0;
		while (!q.empty()) {
			LayerPtr layer = q.front();
			q.pop_front();
			layer->setId(index++);
			layer->setVisited(true);
			for (LayerPtr child : layer->getChildren()) {
				if (child->visitedDependencies()) {
					q.push_back(child);
				}
			}
		}
		layers = order;
		order.clear();
		for (NeuralLayerPtr layer : layers) {
			layer->setVisited(false);
			if (layer->isLeaf()) {
				leafs.push_back(layer);
			}
		}
		*/

	}
};
}

#endif /* INCLUDE_NEURALNETWORK_H_ */
