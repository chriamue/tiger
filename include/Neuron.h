/*
 * Neuron.h
 *
 *  Created on: Jul 2, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURON_H_
#define INCLUDE_NEURON_H_
#include <AlloyMath.h>
namespace tgr {
struct Neuron {
	std::vector<float*> input;
	std::vector<float*> weights;
	float* bias;
	float* output;
	bool active;
	void clear() {
		input.clear();
		weights.clear();
		bias = nullptr;
		output = nullptr;
		active = false;
	}
	Neuron() : bias(nullptr), output(nullptr), active(false) {

	}
};
class NeuronVolume {
public:
	int rows, cols, slices;
	std::vector<Neuron> data;
	NeuronVolume(int rows = 0, int cols = 0, int slices = 0);
	Neuron& operator()(int i, int j, int k);
	const Neuron& operator()(int i, int j, int k) const;

	inline const Neuron& operator ()(aly::int3 ijk) const {
		return operator()(ijk.x, ijk.y, ijk.z);
	}
	inline Neuron& operator ()(aly::int3 ijk) {
		return operator()(ijk.x, ijk.y, ijk.z);
	}
	void resize(int r, int c, int s);
	inline void resize(aly::int3 d) {
		resize(d.x, d.y, d.z);
	}
	void clear();

};
}

#endif /* INCLUDE_NEURON_H_ */
