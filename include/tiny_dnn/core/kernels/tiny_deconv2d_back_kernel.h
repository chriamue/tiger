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

inline void tiny_deconv2d_back_kernel(const deconv_params &params,
                                      const tensor_t &prev_out,
                                      const vec_t &W,
                                      tensor_t &dW,
                                      tensor_t &db,
                                      tensor_t &curr_delta,
                                      tensor_t *prev_delta) {
  // propagate delta to previous layer
  for_i(prev_out.size(), [&](int sample) {
    for (serial_size_t inc = 0; inc < params.in.depth; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        serial_size_t idx = 0;
        idx               = params.in.depth * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out_unpadded.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta[sample][idx];

        idx                 = params.in.get_index(0, 0, inc);
        float_t *pdelta_dst = &(*prev_delta)[sample][idx];

        for (serial_size_t y = 0; y < params.in.height; y++) {
          for (serial_size_t x = 0; x < params.in.width; x++) {
            const float_t *ppw = pw;

            float_t *ppdelta_dst = pdelta_dst + y * params.in.width + x;
            float_t sum{0};

            for (serial_size_t wy = 0; wy < params.weight.height; wy++) {
              for (serial_size_t wx = 0; wx < params.weight.width; wx++) {
                idx = (y * params.h_stride + wy) * params.out.width +
                      (x * params.w_stride + wx);
                sum += ppw[wy * params.weight.width + wx] * pdelta_src[idx];
              }
            }
            *ppdelta_dst += sum;
          }
        }
      }
    }

    // accumulate dw
    for (serial_size_t inc = 0; inc < params.in.depth; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;
        for (serial_size_t wy = 0; wy < params.weight.height; wy++) {
          for (serial_size_t wx = 0; wx < params.weight.width; wx++) {
            float_t dst{0};
            serial_size_t idx    = 0;
            idx                  = params.in.get_index(0, 0, inc);
            const float_t *prevo = &prev_out[sample][idx];

            idx                  = params.out.get_index(wx, wy, outc);
            const float_t *delta = &curr_delta[sample][idx];

            for (serial_size_t y = 0; y < params.in.height; y++) {
              dst +=
                vectorize::dot(prevo + y * params.in.width,
                               delta + y * params.out.width, params.in.width);
            }

            idx = params.in.depth * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      // vec_t& db = *in_grad[2];

      for (serial_size_t outc = 0; outc < params.out.depth; outc++) {
        serial_size_t idx     = params.out.get_index(0, 0, outc);
        const float_t *delta  = &curr_delta[sample][idx];
        const float_t *deltaa = delta + params.out.width * params.out.height;
        db[sample][outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
