/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/product.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
class quantized_fully_connected_layer : public layer {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   **/
  quantized_fully_connected_layer(
    serial_size_t in_dim,
    serial_size_t out_dim,
    bool has_bias          = true,
    backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
  }

  // move constructor
  quantized_fully_connected_layer(quantized_fully_connected_layer &&other)
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_backend(core::backend_t::internal);
  }

  serial_size_t fan_in_size() const override { return params_.in_size; }

  serial_size_t fan_out_size() const override { return params_.out_size; }

  std::vector<index3d<serial_size_t>> in_shape() const override {
    if (params_.has_bias) {
      return {index3d<serial_size_t>(params_.in_size, 1, 1),
              index3d<serial_size_t>(params_.in_size, params_.out_size, 1),
              index3d<serial_size_t>(params_.out_size, 1, 1)};
    } else {
      return {index3d<serial_size_t>(params_.in_size, 1, 1),
              index3d<serial_size_t>(params_.in_size, params_.out_size, 1)};
    }
  }

  std::vector<index3d<serial_size_t>> out_shape() const override {
    return {index3d<serial_size_t>(params_.out_size, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    if (in_data.size() == 2 || in_data.size() == 3) {
      layer::backend_->fully_q(in_data, out_data);

    } else if (in_data.size() == 4 || in_data.size() == 6) {
      layer::backend_->fully_eq(in_data, out_data);
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    layer::backend_->fully_q(in_data, out_data, out_grad, in_grad);
  }

  std::string layer_type() const override { return "q_fully-connected"; }

  friend struct serialization_buddy;

 protected:
  fully_params params_;

  void set_params(const serial_size_t in_size,
                  const serial_size_t out_size,
                  bool has_bias) {
    params_.in_size  = in_size;
    params_.out_size = out_size;
    params_.has_bias = has_bias;
  }

  void init_backend(backend_t backend_type) {
    std::shared_ptr<core::backend> backend = nullptr;

    // allocate new backend
    if (backend_type == backend_t::internal) {
      backend = std::make_shared<core::tiny_backend>(&params_);
    } else {
      throw nn_error("Not supported backend type.");
    }

    if (backend) {
      layer::set_backend(backend);
      layer::set_backend_type(backend_type);
      layer::backend_->set_layer(this);
    } else {
      throw nn_error("Could not allocate the backend.");
    }
  }
};

}  // namespace tiny_dnn
