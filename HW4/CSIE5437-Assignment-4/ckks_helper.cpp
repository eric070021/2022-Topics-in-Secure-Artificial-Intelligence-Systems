#include "ckks_helper.h"

ScalingTechnique rescaleTech;

CKKSHelper::CKKSHelper() { CKKSHelper(3); }

CKKSHelper::CKKSHelper(int depth) : depth(depth) {
  context = get_context(depth);
}

CryptoContext<DCRTPoly> CKKSHelper::get_context(int computation_depth) {
  // Reference:
  //  https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-ckks-bootstrapping.cpp
  //==================================================================================
  // BSD 2-Clause License
  //
  // Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
  //
  // All rights reserved.
  //
  // Author TPOC: contact@openfhe.org
  //==================================================================================

  // Step 1: Set CryptoContext
  CCParams<CryptoContextCKKSRNS> parameters;

  // A. Specify main parameters
  /*  A1) Secret key distribution
   * The secret key distribution for CKKS should either be SPARSE_TERNARY or
   * UNIFORM_TERNARY. The SPARSE_TERNARY distribution was used in the original
   * CKKS paper, but in this example, we use UNIFORM_TERNARY because this is
   * included in the homomorphic encryption standard.
   */
  SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
  parameters.SetSecretKeyDist(secretKeyDist);

  /*  A2) Desired security level based on FHE standards.
   * In this example, we use the "NotSet" option, so the example can run more
   * quickly with a smaller ring dimension. Note that this should be used only
   * in non-production environments, or by experts who understand the security
   * implications of their choices. In production-like environments, we
   * recommend using HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic
   * for 128-bit, 192-bit, or 256-bit security, respectively. If you choose one
   * of these as your security level, you do not need to set the ring dimension.
   */
  parameters.SetSecurityLevel(HEStd_NotSet);
  parameters.SetRingDim(1 << 12);

  /*  A3) Key switching parameters.
   * By default, we use HYBRID key switching with a digit size of 3.
   * Choosing a larger digit size can reduce complexity, but the size of keys
   * will increase. Note that you can leave these lines of code out completely,
   * since these are the default values.
   */
  parameters.SetNumLargeDigits(3);
  parameters.SetKeySwitchTechnique(HYBRID);

  /*  A4) Scaling parameters.
   * By default, we set the modulus sizes and rescaling technique to the
   * following values to obtain a good precision and performance tradeoff. We
   * recommend keeping the parameters below unless you are an FHE expert.
   */
#if NATIVEINT == 128
  // Currently, only FIXEDMANUAL and FIXEDAUTO modes are supported for 128-bit
  // CKKS bootstrapping.
  rescaleTech = FIXEDAUTO;
  usint dcrtBits = 78;
  usint firstMod = 89;
#else
  // All modes are supported for 64-bit CKKS bootstrapping.
  rescaleTech = FLEXIBLEAUTO;
  usint dcrtBits = 59;
  usint firstMod = 60;
#endif

  parameters.SetScalingModSize(dcrtBits);
  parameters.SetScalingTechnique(rescaleTech);
  parameters.SetFirstModSize(firstMod);

  /*  A4) Bootstrapping parameters.
   * We set a budget for the number of levels we can consume in bootstrapping
   * for encoding and decoding, respectively. Using larger numbers of levels
   * reduces the complexity and number of rotation keys, but increases the depth
   * required for bootstrapping. We must choose values smaller than
   * ceil(log2(slots)). A level budget of {4, 4} is good for higher ring
   * dimensions (65536 and higher).
   */
  std::vector<uint32_t> levelBudget = {3, 3};

  // We approximate the number of levels bootstrapping will consume to help set
  // our initial multiplicative depth.
  uint32_t approxBootstrapDepth = 9;

  /* We give the user the option of configuring values for an optimization
   * algorithm in bootstrapping. Here, we specify the giant step for the
   * baby-step-giant-step algorithm in linear transforms for encoding and
   * decoding, respectively. Either choose this to be a power of 2 or an exact
   * divisor of the number of slots. Setting it to have the default value of {0,
   * 0} allows OpenFHE to choose the values automatically.
   */
  std::vector<uint32_t> bsgsDim = {0, 0};

  /*  A5) Multiplicative depth.
   * The goal of bootstrapping is to increase the number of available levels we
   * have, or in other words, to dynamically increase the multiplicative depth.
   * However, the bootstrapping procedure itself needs to consume a few levels
   * to run. We compute the number of bootstrapping levels required using
   * GetBootstrapDepth, and add it to levelsUsedBeforeBootstrap to set our
   * initial multiplicative depth.
   */
  uint32_t levelsUsedBeforeBootstrap = computation_depth;
  usint totalDepth = levelsUsedBeforeBootstrap +
                     FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth,
                                                   levelBudget, secretKeyDist);
  parameters.SetMultiplicativeDepth(totalDepth);

  // Generate crypto context.
  CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

  // Enable features that you wish to use. Note, we must enable FHE to use
  // bootstrapping.
  cryptoContext->Enable(PKE);
  cryptoContext->Enable(KEYSWITCH);
  cryptoContext->Enable(LEVELEDSHE);
  cryptoContext->Enable(ADVANCEDSHE);
  cryptoContext->Enable(FHE);

  ring_dim = cryptoContext->GetRingDimension();
  n_slots = ring_dim / 2;
  std::cout << "CKKS scheme is using ring dimension " << ring_dim << std::endl
            << std::endl;

  // Step 2: Precomputations for bootstrapping
  cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, n_slots);

  // Step 3: Key Generation
  key_pair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(key_pair.secretKey);
  // Generate bootstrapping keys.
  cryptoContext->EvalBootstrapKeyGen(key_pair.secretKey, n_slots);
  // Generate rotation keys.
  cryptoContext->EvalRotateKeyGen(key_pair.secretKey, {1, -1});

  return cryptoContext;
}

Ciphertext<DCRTPoly> CKKSHelper::encrypt(const vector<T> &x) {
  // from message to plaintext
  vector<double> value(x.begin(), x.end());
  Plaintext ptxt = context->MakeCKKSPackedPlaintext(value, 1, 0);
  ptxt->SetLength(x.size());

  // encrypt
  Ciphertext<DCRTPoly> ciph = context->Encrypt(key_pair.publicKey, ptxt);

  return ciph;
}

vector<T> CKKSHelper::decrypt(const Ciphertext<DCRTPoly> &ciph, int length) {
  Plaintext result;
  context->Decrypt(key_pair.secretKey, ciph, &result);
  result->SetLength(length);
  vector<double> decoded = result->GetRealPackedValue();
  return vector<T>(decoded.begin(), decoded.end());
}

// TODO: please finish this function
Ciphertext<DCRTPoly> CKKSHelper::mat_vec_mul(const vector<vector<T>> &M,
                                             const Ciphertext<DCRTPoly> &ctx) {
  assert(M.size());
  assert(M[0].size());

  // build diagonal matrix
  int n = n_slots;
  vector<vector<double>> diagonal(n, vector<double>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i >= (int)M.size() || j >= (int)M[0].size())
        continue;
      // TODO: construct the diagonal matrix according to M
      diagonal[j][i] = M[i][(i + j) % (int)M[0].size()];
    }
  }

  Ciphertext<DCRTPoly> v = ctx;
  Ciphertext<DCRTPoly> sum, cur;
  Plaintext ptxt, ptxtsum;
  vector<double> sum_vec(n);
  ptxtsum = context->MakeCKKSPackedPlaintext(sum_vec);
  sum = context->Encrypt(key_pair.publicKey, ptxtsum);

  // make v = [v,v] because of rotation
  Ciphertext<DCRTPoly> temp = ctx;
  for(int j = 0; j < (int)M[0].size(); j++){
    temp = context->EvalRotate(temp, -1);
  }
  v = context->EvalAdd(v, temp);

  for (int i = 0; i < n; i++) {
    // if (i >= (int)M[0].size())
    //   break;
    // TODO: element-wise multiplication (cur = v * diagonal[i])
    ptxt = context->MakeCKKSPackedPlaintext(diagonal[i]);
    cur = context->EvalMult(v, ptxt);
    // TODO: element-wise addition (sum = sum + cur)
    sum = context->EvalAdd(sum, cur);
    // TODO: rotate ciphertext (left_rotate(v, 1))
    v = context->EvalRotate(v, 1);
  }

  return sum;
}

Ciphertext<DCRTPoly> CKKSHelper::vec_add(const vector<T> &v,
                                         const Ciphertext<DCRTPoly> &ctx) {
  vector<double> _v(v.begin(), v.end());
  Plaintext ptxt = context->MakeCKKSPackedPlaintext(_v);
  return context->EvalAdd(ptxt, ctx);
}

Ciphertext<DCRTPoly> CKKSHelper::vec_mul(const Ciphertext<DCRTPoly> &ctx1,
                                         const Ciphertext<DCRTPoly> &ctx2) {
  return context->EvalMult(ctx1, ctx2);
}

Ciphertext<DCRTPoly> CKKSHelper::bootstrap(const Ciphertext<DCRTPoly> &ctx) {
  return context->EvalBootstrap(ctx);
}
