/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_internal(const tensor_t &in_data,
                               const vec_t &W,
                               const vec_t &bias,
                               tensor_t &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  for_(parallelize, 0, in_data.size(),
       [&](const blocked_range &r) {
         size_t out_area           = params.out.area();
         serial_size_t iw          = params.in_padded.width;
         serial_size_t id          = params.in.depth;
         serial_size_t ow          = params.out.width;
         serial_size_t oh          = params.out.height;
         serial_size_t od          = params.out.depth;
         serial_size_t kw          = params.weight.width;
         serial_size_t kh          = params.weight.height;
         serial_size_t elem_stride = params.w_stride;
         serial_size_t line_stride = iw * params.h_stride;
         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &in = in_data[sample];
           vec_t &a        = out_data[sample];
           for (serial_size_t o = 0; o < od; o++) {
             float_t *pa = &a[params.out.get_index(0, 0, o)];
             for (serial_size_t inc = 0; inc < id; inc++) {
               if (!params.tbl.isConnected(o, inc)) continue;
               serial_size_t idx;
               idx                = params.weight.get_index(0, 0, id * o + inc);
               const float_t *pw  = &W[idx];
               idx                = params.in_padded.get_index(0, 0, inc);
               const float_t *pin = &in[idx];
               float_t *pout      = pa;
               for (serial_size_t y = 0; y < oh; y++) {
                 const float_t *pin_line = pin;
                 for (serial_size_t x = 0; x < ow; x++) {
                   const float_t *pin_element = pin_line;
                   const float_t *pw_element  = pw;
                   float_t sum{0};
                   // should be optimized for small kernel(3x3,5x5)
                   for (serial_size_t wy = 0; wy < kh; wy++) {    // NOLINT
                     for (serial_size_t wx = 0; wx < kw; wx++) {  // NOLINT
                       sum += pw_element[wx] * pin_element[wx];
                     }
                     pw_element += kw;
                     pin_element += iw;
                   }
                   pout[x] += sum;
                   pin_line += elem_stride;
                 }
                 pout += ow;
                 pin += line_stride;
               }
             }
             if (params.has_bias) {
               vectorize::add(bias[o], out_area, pa);
             }
           }
         }
       },
       0);
}

/******************************************************************/

template <typename tensor_t, typename vec_t>
void conv2d_op_internal(const tensor_t &prev_out,
                        const vec_t &W,
                        tensor_t &dW,
                        tensor_t &db,
                        tensor_t &curr_delta,
                        tensor_t &prev_delta,
                        const core::conv_params &params,
                        const bool parallelize) {
  typedef typename vec_t::value_type float_t;

  for_i(parallelize, prev_out.size(), [&](int sample) {
    // propagate delta to previous layer
    for (serial_size_t inc = 0; inc < params.in.depth; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth; outc++) {
        if (!params.tbl.isConnected(outc, inc)) continue;

        serial_size_t idx = 0;
        idx               = params.in.depth * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_delta[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t* pdelta_dst = &(*prev_delta)[sample][idx];
        float_t *pdelta_dst = &prev_delta[sample][idx];

        for (serial_size_t y = 0; y < params.out.height; y++) {
          for (serial_size_t x = 0; x < params.out.width; x++) {
            const float_t *ppw = pw;

            idx                       = y * params.out.width + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width +
              x * params.w_stride;

            for (serial_size_t wy = 0; wy < params.weight.height;
                 wy++) {  // NOLINT
              for (serial_size_t wx = 0; wx < params.weight.width;
                   wx++) {  // NOLINT
                idx = wy * params.in_padded.width + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw
    for (serial_size_t inc = 0; inc < params.in.depth; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth; outc++) {
        if (!params.tbl.isConnected(outc, inc)) continue;

        for (serial_size_t wy = 0; wy < params.weight.height; wy++) {
          for (serial_size_t wx = 0; wx < params.weight.width; wx++) {
            float_t dst{0};

            serial_size_t idx    = 0;
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev_out[sample][idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr_delta[sample][idx];

            if (params.w_stride > 1) {
              for (serial_size_t y = 0; y < params.out.height; y++) {
                serial_size_t prevo_idx =
                  y * params.in_padded.width * params.h_stride;
                serial_size_t delta_idx = y * params.out.width;

                for (serial_size_t x = 0; x < params.out.width; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              for (serial_size_t y = 0; y < params.out.height; y++) {
                dst += vectorize::dot(
                  prevo + y * params.in_padded.width * params.h_stride,
                  delta + y * params.out.width, params.out.width);
              }
            }

            idx = params.in.depth * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
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
}  // namespace tiny_dnn
