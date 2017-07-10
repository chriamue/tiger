#include "MNIST.h"
namespace tgr {
void parse_mnist_header(std::ifstream& ifs, mnist_header& header) {
	ifs.read((char*) &header.magic_number, 4);
	ifs.read((char*) &header.num_items, 4);
	ifs.read((char*) &header.num_rows, 4);
	ifs.read((char*) &header.num_cols, 4);

	if (is_little_endian()) {
		reverse_endian(&header.magic_number);
		reverse_endian(&header.num_items);
		reverse_endian(&header.num_rows);
		reverse_endian(&header.num_cols);
	}

	if (header.magic_number != 0x00000803 || header.num_items <= 0)
		throw std::runtime_error("MNIST label-file format error");
	if (ifs.fail() || ifs.bad())
		throw std::runtime_error("file error");
}
void parse_mnist_image(std::ifstream& ifs, const mnist_header& header,
		float scale_min, float scale_max, int x_padding, int y_padding,
		aly::Image1f& dst) {
	const int width = header.num_cols + 2 * x_padding;
	const int height = header.num_rows + 2 * y_padding;
	std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);
	ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);
	dst.resize(width, height);
	for (uint32_t y = 0; y < header.num_rows; y++)
		for (uint32_t x = 0; x < header.num_cols; x++)
			dst[width * (y + y_padding) + x + x_padding] = aly::float1(
					(image_vec[y * header.num_cols + x] / 255.0f)
							* (scale_max - scale_min) + scale_min);
}
void parse_mnist_labels(const std::string& label_file,
		std::vector<int>& labels) {
	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
	labels.clear();
	if (ifs.bad() || ifs.fail())
		throw std::runtime_error("failed to open file:" + label_file);

	uint32_t magic_number, num_items;

	ifs.read((char*) &magic_number, 4);
	ifs.read((char*) &num_items, 4);

	if (is_little_endian()) { // MNIST data is big-endian format
		reverse_endian(&magic_number);
		reverse_endian(&num_items);
	}
	if (magic_number != 0x00000801 || num_items <= 0)
		throw std::runtime_error("MNIST label-file format error");
	labels.reserve(num_items);
	for (uint32_t i = 0; i < num_items; i++) {
		uint8_t label;
		ifs.read((char*) &label, 1);
		labels.push_back(label);
	}
}

void parse_mnist_images(const std::string& image_file,
		std::vector<Tensor>& images, float scale_min, float scale_max,
		int x_padding, int y_padding) {

	if (x_padding < 0 || y_padding < 0)
		throw std::runtime_error("padding size must not be negative");
	if (scale_min >= scale_max)
		throw std::runtime_error("scale_max must be greater than scale_min");

	std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.bad() || ifs.fail())
		throw std::runtime_error("failed to open file:" + image_file);

	mnist_header header;

	parse_mnist_header(ifs, header);
	images.clear();
	images.reserve(header.num_items);
	for (uint32_t i = 0; i < header.num_items; i++) {
		aly::Image1f image;
		parse_mnist_image(ifs, header, scale_min, scale_max, x_padding,y_padding, image);
		Tensor t = std::vector<Storage> {Storage(image.data.begin(),image.data.end()) };
		images.push_back(t);
	}
}
}
