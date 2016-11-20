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
#ifndef NeuralLayer_H_
#define NeuralLayer_H_
#include <AlloyMath.h>
#include <AlloyContext.h>
#include "Neuron.h"
#include <vector>
#include <set>
namespace tgr {
	class NeuralLayer {
		protected:
			std::vector<Neuron> neurons;
			int id;
		public:
			int width;
			int height;
			int bins;

			typedef Neuron ValueType;
			typedef typename std::vector<ValueType>::iterator iterator;
			typedef typename std::vector<ValueType>::const_iterator const_iterator;
			typedef typename std::vector<ValueType>::reverse_iterator reverse_iterator;
			iterator begin() {
				return neurons.begin();
			}
			iterator end() {
				return neurons.end();
			}
			void setId(int i) {
				id = i;
			}
			int getId()const {
				return id;
			}
			void setFunction(const NeuronFunction& func);
			int getBin(size_t index) const;
			int getBin(const Neuron& n) const;

			aly::int2 dimensions() const {
				return aly::int2(width, height);
			}
			float getAspect() const {
				return width / (float)height;
			}
			Neuron& get(int i, int j);
			const Neuron& get(int i, int j) const;

			void resize(int width, int height, int b=1);
			const Neuron& operator[](const size_t i) const;
			Neuron& operator[](const size_t i);
			Neuron& operator()(const int i, const int j);
			Neuron& operator()(const size_t i, const size_t j);
			Neuron& operator()(const aly::int2 ij);
			Neuron& operator()(const Terminal ij);
			const Neuron& operator()(const int i, const int j) const;
			const Neuron& operator()(const size_t i, const size_t j) const;
			const Neuron& operator()(const aly::int2 ij) const;
			const Neuron& operator()(const Terminal ij) const;
			void draw(aly::AlloyContext* context, const aly::box2px& bounds);
			NeuralLayer(int width=0,int height=0,int bins=1,int id = 0);
	};
	typedef std::shared_ptr<NeuralLayer> NeuralLayerPtr;
}
#endif