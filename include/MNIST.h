/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _MNIST_H_
#define _MNIST_H_
#include <fstream>
#include <cstdint>
#include <AlloyImage.h>
#include <NeuralSignal.h>
namespace tgr {
	struct mnist_header {
		uint32_t magic_number;
		uint32_t num_items;
		uint32_t num_rows;
		uint32_t num_cols;
	};
	template<typename T>
	T* reverse_endian(T* p) {
		std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
		return p;
	}

	inline bool is_little_endian() {
		int x = 1;
		return *(char*)&x != 0;
	}
	void parse_mnist_header(std::ifstream& ifs, mnist_header& header);
	void parse_mnist_image(std::ifstream& ifs,
		const mnist_header& header,
		float scale_min,
		float scale_max,
		int x_padding,
		int y_padding,
		aly::Image1f& dst);

	/**
	 * parse MNIST database format labels with rescaling/resizing
	 * http://yann.lecun.com/exdb/mnist/
	 *
	 * @param label_file [in]  filename of database (i.e.train-labels-idx1-ubyte)
	 * @param labels     [out] parsed label data
	 **/
	void parse_mnist_labels(const std::string& label_file, std::vector<int>& labels);
	/**
	 * parse MNIST database format images with rescaling/resizing
	 * http://yann.lecun.com/exdb/mnist/
	 * - if original image size is WxH, output size is (W+2*x_padding)x(H+2*y_padding)
	 * - extra padding pixels are filled with scale_min
	 *
	 * @param image_file [in]  filename of database (i.e.train-images-idx3-ubyte)
	 * @param images     [out] parsed image data
	 * @param scale_min  [in]  min-value of output
	 * @param scale_max  [in]  max-value of output
	 * @param x_padding  [in]  adding border width (left,right)
	 * @param y_padding  [in]  adding border width (top,bottom)
	 *
	 * [example]
	 * scale_min=-1.0, scale_max=1.0, x_padding=1, y_padding=0
	 *
	 * [input]       [output]
	 *  64  64  64   -1.0 -0.5 -0.5 -0.5 -1.0
	 * 128 128 128   -1.0  0.0  0.0  0.0 -1.0
	 * 255 255 255   -1.0  1.0  1.0  1.0 -1.0
	 *
	 **/
	void parse_mnist_images(const std::string& image_file,
		std::vector<Tensor>& images,
		float scale_min = 0.0f,
		float scale_max = 1.0f,
		int x_padding = 0,
		int y_padding = 0);
}
#endif
