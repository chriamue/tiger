/*
 * NeuralParams.h
 *
 *  Created on: Jun 24, 2017
 *      Author: blake
 */

#ifndef INCLUDE_NEURALPARAMS_H_
#define INCLUDE_NEURALPARAMS_H_

#include <AlloyImage.h>
#include <NeuralSignal.h>
class conv_params;
class fully_params;
class maxpool_params;
class global_avepool_params;

/* Base class to model operation parameters */
class Params {
 public:
  Params() {}

  conv_params &conv();
  fully_params &fully();
  maxpool_params &maxpool();
  global_avepool_params &global_avepool();
};


inline conv_params &Params::conv() {
  return *(static_cast<conv_params *>(this));
}

class fully_params : public Params {
 public:
  int in_size_;
  int out_size_;
  bool has_bias_;
};

// TODO(nyanp): can we do better here?
inline fully_params &Params::fully() {
  return *(static_cast<fully_params *>(this));
}

class maxpool_params : public Params {
 public:
  aly::int3 in;
  aly::int3 out;
  int pool_size_x;
  int pool_size_y;
  int stride_x;
  int stride_y;
  Padding pad_type;

  /* mapping out => max_index(in) (1:1) */
  std::vector<std::vector<int>> out2inmax;
  /* mapping out => in (1:N) */
  std::vector<std::vector<int>> out2in;
  /* mapping in => out (N:1) */
  std::vector<int> in2out;
};

struct max_pooling_layer_worker_specific_storage {
  /* mapping out => max_index(in) (1:1) */
  std::vector<std::vector<int>> out2inmax_;
};

// TODO(nyanp): can we do better here?
inline maxpool_params &Params::maxpool() {
  return *(static_cast<maxpool_params *>(this));
}


class global_avepool_params : public Params {
 public:
  shape3d in;
  shape3d out;
};

inline global_avepool_params &Params::global_avepool() {
  return *(static_cast<global_avepool_params *>(this));
}




#endif /* INCLUDE_NEURALPARAMS_H_ */
