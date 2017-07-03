/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_deconv2d_kernel(const deconv_params &params,
                                 const tensor_t &in,
                                 const vec_t &W,
                                 const vec_t &bias,
                                 tensor_t &out,
                                 const bool layer_parallelize) {
  for_i(layer_parallelize, in.size(), [&](int sample) {
    for (serial_size_t o = 0; o < params.out.depth; o++) {
      for (serial_size_t inc = 0; inc < params.in.depth; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        serial_size_t idx = 0;
        idx               = params.in.depth * o + inc;
        idx               = params.weight.get_index(0, 0, idx);
        assert(idx < W.size());
        const float_t *pw = &W[idx];

        idx = params.in.get_index(0, 0, inc);
        assert(static_cast<serial_size_t>(sample) < in.size() &&
               idx <= in[sample].size());
        const float_t *pi = &in[sample][idx];

        idx = params.out.get_index(0, 0, o);
        assert(static_cast<serial_size_t>(sample) < out.size() &&
               idx <= out[sample].size());
        float_t *pout = &out[sample][idx];

        for (serial_size_t y = 0; y < params.in.height; y++) {
          for (serial_size_t x = 0; x < params.in.width; x++) {
            const float_t *ppw = pw;
            const float_t *ppi = pi + y * params.in.width + x;
            // should be optimized for small kernel(3x3,5x5)
            for (serial_size_t wy = 0; wy < params.weight.height; wy++) {
              for (serial_size_t wx = 0; wx < params.weight.width; wx++) {
                pout[(y * params.h_stride + wy) * params.out.width +
                     (x * params.w_stride + wx)] +=
                  ppw[wy * params.weight.width + wx] * (*ppi);
              }
            }
          }
        }
      }

      if (params.has_bias) {
        float_t *pout  = &out[sample][params.out.get_index(0, 0, o)];
        float_t *pout2 = pout + params.out.width * params.out.height;
        std::for_each(pout, pout2, [&](float_t &f) { f += bias[o]; });
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
