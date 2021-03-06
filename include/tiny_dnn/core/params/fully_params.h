/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <tiny_dnn/core/params/params.h>

namespace tiny_dnn {
namespace core {

class fully_params : public Params {
 public:
  serial_size_t in_size;
  serial_size_t out_size;
  bool has_bias;
};

// TODO(nyanp): can we do better here?
inline fully_params &Params::fully() {
  return *(static_cast<fully_params *>(this));
}

}  // namespace core
}  // namespace tiny_dnn
