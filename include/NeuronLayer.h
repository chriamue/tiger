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
#ifndef NEURONLAYER_H_
#define NEURONLAYER_H_
#include <AlloyMath.h>
#include "Neuron.h"
#include <vector>
#include <set>
namespace tgr {
	class NeuronLayer {
		protected:
			std::vector<Neuron> neurons;
			std::vector<size_t> active;
			int id;
		public:
			int width;
			int height;
			int bins;
			void setId(int i) {
				id = i;
			}
			int getId()const {
				return id;
			}
			void resize(int r, int c, int s);
			const Neuron& operator[](const size_t i) const;
			Neuron& operator[](const size_t i);
			Neuron& operator()(const int i, const int j, const int k);
			Neuron& operator()(const size_t i, const size_t j, const size_t k);
			Neuron& operator()(const aly::int3 ijk);
			Neuron& operator()(const Terminal ijk);
			const Neuron& operator()(const int i, const int j, const int k) const;
			const Neuron& operator()(const size_t i, const size_t j, const size_t k) const;
			const Neuron& operator()(const aly::int3 ijk) const;
			const Neuron& operator()(const Terminal ijk) const;
			NeuronLayer(int width,int height,int bins=1,int id = 0);
	};
	typedef std::shared_ptr<NeuronLayer> NeuronLayerPtr;
}
#endif