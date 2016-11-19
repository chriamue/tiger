/*
* Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
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
#ifndef NEURON_H_
#define NEURON_H_
#include <AlloyMath.h>
#include "NeuronFunction.h"

namespace tgr {
	struct NeuronIndex:aly::int2 {
		int layer;
		int bin;
		NeuronIndex(int x=0, int y=0, int b=0, int l=-1) :aly::int2(x, y), bin(b), layer(l) {

		}
	};
	struct Axon {
		NeuronIndex index;
		float weight;
		float delta;
		Axon() :index(0, 0), weight(0.0f), delta(0.0f) {

		}
	};
	typedef std::shared_ptr<Axon> AxonPtr;
	class Neuron {
	protected:
		NeuronFunction neuronFunction;
		std::vector<AxonPtr> inputs;
		AxonPtr output;
	public:
		float value;
		Neuron(float val=0.0f);
		void setResponseFunction(const NeuronFunction& func) {
			neuronFunction = func;
		}
	};
}
#endif