#ifndef _NEURAL_TENSOR_H_
#define _NEURAL_TENSOR_H_
#include <AlloyMath.h>
#include <AlloyOptimizationMath.h>
namespace aly {
	//Borrowed from Stan Melax ...
	aly::int2 make_packed_stride(const aly::int2 & dims) { return{ 1, dims.x }; }
	aly::int3 make_packed_stride(const aly::int3 & dims) { return{ 1, dims.x, dims.x*dims.y }; }
	aly::int4 make_packed_stride(const aly::int4 & dims) { return{ 1, dims.x, dims.x*dims.y, dims.x*dims.y*dims.z }; }
	template<class T, int K> struct tensorview // Note: Works for K in {2,3,4}
	{
		using               intK = aly::vec<int, K>;
		T*					data;
		intK                dims, stride;
		tensorview(T* data, intK dims, intK stride) : data(data), dims(dims), stride(stride) {}
		tensorview(T* data, intK dims) : tensorview(data, dims, make_packed_stride(dims)) {}
		template<class U>   tensorview(const tensorview<U, K> & view) : tensorview(view.data, view.dims, view.stride) {} // Allows for T -> const T and other such conversions
		T &             operator[] (intK i) const {
			return data[dot(stride, i)];
		}
		tensorview<T,K - 1>   operator[] (int i) const {
			return{ data + stride[K - 1] * i, (const aly::vec<int,K - 1> &)dims, (const aly::vec<int,K - 1> &)stride };
		}
		tensorview subview(intK woffset, intK wdims) const {
			return { data + dot(stride, woffset), wdims, stride };
		}
	};
	typedef tensorview<float, 1> tensorview1f;
	typedef tensorview<float, 2> tensorview2f;
	typedef tensorview<float, 3> tensorview3f;
	typedef tensorview<float, 4> tensorview4f;
	typedef tensorview<double, 1> tensorview1d;
	typedef tensorview<double, 2> tensorview2d;
	typedef tensorview<double, 3> tensorview3d;
	typedef tensorview<double, 4> tensorview4d;
}

#endif