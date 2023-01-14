#ifndef CKKS_HELPER_H
#define CKKS_HELPER_H
#include "params.h"
#include "utils.h"

class CKKSHelper {
public:
  CKKSHelper();
  CKKSHelper(int);

  Ciphertext<DCRTPoly> encrypt(const vector<T> &);
  vector<T> decrypt(const Ciphertext<DCRTPoly> &, int);
  Ciphertext<DCRTPoly> mat_vec_mul(const vector<vector<T>> &M,
                                   const Ciphertext<DCRTPoly> &ctx);
  Ciphertext<DCRTPoly> vec_add(const vector<T> &v,
                               const Ciphertext<DCRTPoly> &ctx);
  Ciphertext<DCRTPoly> vec_mul(const Ciphertext<DCRTPoly> &ctx1,
                               const Ciphertext<DCRTPoly> &ctx2);
  Ciphertext<DCRTPoly> bootstrap(const Ciphertext<DCRTPoly> &ctx);

private:
  CryptoContext<DCRTPoly> get_context(int);
  CryptoContext<DCRTPoly> context;
  KeyPair<DCRTPoly> key_pair;
  int depth;
  int n_slots;
  int ring_dim;
};

#endif
