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

#include "TigerApp.h"
int main(int argc, char *argv[]) {
	int index = -1;
	try {
		if (argc == 1) {
			std::cout << "Usage: " << argv[0] << " [example index]\nToy Examples:" << std::endl;
			std::cout << "[0] XOR" << std::endl;
			std::cout << "[1] Waves" << std::endl;
			std::cout << "[2] LeNET5" << std::endl;
			std::cout << ">> Enter Example Number: ";
			std::cin >> index;
		} else {
			index = std::atoi(argv[1]);
		}
		TigerApp flowPane(index);
		flowPane.run(1);
	} catch (std::exception& e) {
		std::cout << "Main Error: " << e.what() << std::endl;
		std::getchar();
		return 1;
	}
	return 0;

}
