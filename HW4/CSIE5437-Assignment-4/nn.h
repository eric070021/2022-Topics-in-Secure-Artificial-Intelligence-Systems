#ifndef NN_H
#define NN_H
#include "ckks_helper.h"
#include "params.h"
#include "utils.h"

class Linear {
public:
  Linear() {}
  Linear(string name) : name(name) { load(); }
  vector<T> forward(const vector<T> &x);
  Ciphertext<DCRTPoly> enc_forward(const Ciphertext<DCRTPoly> &x, CKKSHelper &);
  void load();
  string name;
  vector<vector<T>> weight;
  vector<T> bias;
};

#endif
