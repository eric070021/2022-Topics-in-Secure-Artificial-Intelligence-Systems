#include "ckks_helper.h"
#include "json.hpp"
#include "nn.h"
#include "openfhe.h"
#include "params.h"
#include "utils.h"
#include <bits/stdc++.h>
using namespace std;
using namespace lbcrypto;

class Net {
public:
  Net() {
    input_fc = Linear("input_fc");
    hidden_fc = Linear("hidden_fc");
    output_fc = Linear("output_fc");
  };
  vector<T> forward(const vector<T> &input) {
    vector<T> x = input_fc.forward(input);
    x = mul(x, x);
    x = hidden_fc.forward(x);
    x = mul(x, x);
    x = output_fc.forward(x);
    return x;
  }
  // forward with encrypted inputs
  Ciphertext<DCRTPoly> enc_forward(const Ciphertext<DCRTPoly> &x,
                                   CKKSHelper &helper) {
    Ciphertext<DCRTPoly> ret = input_fc.enc_forward(x, helper);
    ret = helper.vec_mul(ret, ret); // activation function

    ret = hidden_fc.enc_forward(ret, helper);
    ret = helper.vec_mul(ret, ret);

    ret = output_fc.enc_forward(ret, helper);
    return ret;
  }
  Linear input_fc;
  Linear hidden_fc;
  Linear output_fc;
};

int main() {
  // prepare data
  vector<vector<T>> images;
  vector<int> labels;
  tie(images, labels) = get_data();

  Net net;
  // a class that initializes the crypto context and manages HE operations
  CKKSHelper helper(5);

  for (int idx = 0; idx < 3; idx++) {
    cout << "=== forwarding image " << idx << " ===" << endl;

    // === normal forward ===
    cout << "| normal forward |" << endl;
    // forward
    vector<T> pred = net.forward(images[idx]);
    cout << "pred: ";
    print(pred);
    // get max index
    int max_idx = get_max_idx(pred);
    cout << "max idx: " << max_idx << endl;

    // === encrypted forward ===
    cout << "| encrypted forward |" << endl;
    // encrypt the input
    Ciphertext<DCRTPoly> enc_image = helper.encrypt(images[idx]);
    // forward
    Ciphertext<DCRTPoly> enc_output = net.enc_forward(enc_image, helper);
    // decrypt the output
    vector<T> enc_pred = helper.decrypt(enc_output, 10);
    cout << "pred: ";
    print(enc_pred);
    // get max index
    int enc_max_idx = get_max_idx(enc_pred);
    cout << "max idx: " << enc_max_idx << endl;

    // === ground truth label ===
    cout << "| ground truth label |" << endl;
    cout << labels[idx] << endl;
  }
}
