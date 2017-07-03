/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <tiny_dnn/core/params/params.h>
#include <tiny_dnn/util/util.h>
namespace tiny_dnn {
namespace core {

struct conv_layer_worker_specific_storage {
  std::vector<const vec_t *> prev_out_padded;
  std::vector<vec_t> prev_out_buf;
  std::vector<vec_t> prev_delta_padded;
};

struct ConnectionTable {
  ConnectionTable() : rows(0), cols(0) {}
  ConnectionTable(const bool *ar, serial_size_t rows, serial_size_t cols)
    : connected(rows * cols), rows(rows), cols(cols) {
    std::copy(ar, ar + rows * cols, connected.begin());
  }
  ConnectionTable(serial_size_t ngroups,
                   serial_size_t rows,
                   serial_size_t cols)
    : connected(rows * cols, false), rows(rows), cols(cols) {
    if (rows % ngroups || cols % ngroups) {
      throw nn_error("invalid group size");
    }
    serial_size_t row_group = rows / ngroups;
    serial_size_t col_group = cols / ngroups;
    serial_size_t idx = 0;
    for (serial_size_t g = 0; g < ngroups; g++) {
      for (serial_size_t r = 0; r < row_group; r++) {
        for (serial_size_t c = 0; c < col_group; c++) {
          idx             = (r + g * row_group) * cols + c + g * col_group;
          connected[idx] = true;
        }
      }
    }
  }
  bool isConnected(serial_size_t x, serial_size_t y) const {
    return isEmpty() ? true : connected[y * cols + x];
  }
  bool isEmpty() const { return rows == 0 && cols == 0; }
  std::deque<bool> connected;
  serial_size_t rows;
  serial_size_t cols;
};

class conv_params : public Params {
 public:
  ConnectionTable tbl;
  index3d<serial_size_t> in;
  index3d<serial_size_t> in_padded;
  index3d<serial_size_t> out;
  index3d<serial_size_t> weight;
  bool has_bias;
  padding pad_type;
  serial_size_t w_stride;
  serial_size_t h_stride;

  friend std::ostream &operator<<(std::ostream &o,
                                  const core::conv_params &param) {
    o << "in:        " << param.in << "\n";
    o << "out:       " << param.out << "\n";
    o << "in_padded: " << param.in_padded << "\n";
    o << "weight:    " << param.weight << "\n";
    o << "has_bias:  " << param.has_bias << "\n";
    o << "w_stride:  " << param.w_stride << "\n";
    o << "h_stride:  " << param.h_stride << "\n";
    return o;
  }
};

inline conv_params &Params::conv() {
  return *(static_cast<conv_params *>(this));
}

class Conv2dPadding {
 public:
  Conv2dPadding() {}
  Conv2dPadding(const conv_params &params) : params_(params) {}

  /* Applies padding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with padding applied
   */
  void copy_and_pad_input(const tensor_t &in, tensor_t &out) {
    if (params_.pad_type == padding::valid) {
      return;
    }

    tensor_t buf(in.size());

    for_i(true, buf.size(), [&](int sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in_padded.size());

      // make padded version in order to avoid corner-case in fprop/bprop
      for (serial_size_t c = 0; c < params_.in.depth; c++) {
        float_t *pimg = &buf[sample][params_.in_padded.get_index(
          params_.weight.width / 2, params_.weight.height / 2, c)];
        const float_t *pin = &in[sample][params_.in.get_index(0, 0, c)];

        for (serial_size_t y = 0; y < params_.in.height; y++) {
          std::copy(pin, pin + params_.in.width, pimg);
          pin += params_.in.width;
          pimg += params_.in_padded.width;
        }
      }
    });

    // shrink buffer to output
    out = buf;
  }

  /* Applies unpadding to an input tensor given the convolution parameters
   *
   * @param in The input tensor
   * @param out The output tensor with unpadding applied
   */
  void copy_and_unpad_delta(const tensor_t &delta, tensor_t &delta_unpadded) {
    if (params_.pad_type == padding::valid) {
      return;
    }

    tensor_t buf(delta.size());

    for_i(true, buf.size(), [&](int sample) {
      // alloc temporary buffer.
      buf[sample].resize(params_.in.size());

      for (serial_size_t c = 0; c < params_.in.depth; c++) {
        const float_t *pin = &delta[sample][params_.in_padded.get_index(
          params_.weight.width / 2, params_.weight.height / 2, c)];
        float_t *pdst = &buf[sample][params_.in.get_index(0, 0, c)];

        for (serial_size_t y = 0; y < params_.in.height; y++) {
          std::copy(pin, pin + params_.in.width, pdst);
          pdst += params_.in.width;
          pin += params_.in_padded.width;
        }
      }
    });

    // shrink buffer to output
    delta_unpadded = buf;
  }

 private:
  conv_params params_;
};

}  // namespace core
}  // namespace tiny_dnn
