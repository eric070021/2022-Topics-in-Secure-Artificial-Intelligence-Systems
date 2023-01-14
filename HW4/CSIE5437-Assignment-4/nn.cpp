#include "nn.h"

vector<T> Linear::forward(const vector<T> &x) {
  vector<T> ret = matmul(weight, x);
  ret = add(ret, bias);
  return ret;
}

void Linear::load() { load_linear(name, weight, bias); }

Ciphertext<DCRTPoly> Linear::enc_forward(const Ciphertext<DCRTPoly> &x,
                                         CKKSHelper &helper) {
  Ciphertext<DCRTPoly> ret = helper.mat_vec_mul(weight, x);
  ret = helper.vec_add(bias, ret);
  return ret;
}
