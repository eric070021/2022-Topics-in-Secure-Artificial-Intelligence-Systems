#include "utils.h"
#include "json.hpp"
#include <bits/stdc++.h>

// === load matrix and array from npy files ===

void load_matrix(const string npy_file, vector<vector<T>> &matrix) {
  vector<T> tmp;
  vector<unsigned long> shape;
  bool is_fortran;
  npy::LoadArrayFromNumpy(npy_file, shape, is_fortran, tmp);
  matrix.resize(shape[0]);
  for (int i = 0; i < (int)shape[0]; i++) {
    for (int j = 0; j < (int)shape[1]; j++)
      matrix[i].push_back(tmp[i * shape[1] + j]);
  }
}

void load_array(const string npy_file, vector<T> &arr) {
  vector<unsigned long> shape;
  bool is_fortran;
  npy::LoadArrayFromNumpy(npy_file, shape, is_fortran, arr);
}

void load_linear(const string linear_name, vector<vector<T>> &weight,
                 vector<T> &bias) {
  load_matrix("npy/" + linear_name + "_weight.npy", weight);
  load_array("npy/" + linear_name + "_bias.npy", bias);
}

// === plaintext operations ===

vector<T> matmul(const vector<vector<T>> &M, const vector<T> &v) {
  assert(M.size());
  assert(M[0].size() == v.size());
  int n = M.size(), m = M[0].size();
  vector<T> ret(n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      ret[i] += M[i][j] * v[j];
    }
  }

  return ret;
}

vector<T> add(const vector<T> &a, const vector<T> &b) {
  assert(a.size() == b.size());
  vector<T> ret(a.size());
  for (int i = 0; i < (int)a.size(); i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

vector<T> add(const vector<T> &a, const T &b) {
  vector<T> ret(a.size());
  for (int i = 0; i < (int)a.size(); i++) {
    ret[i] = a[i] + b;
  }
  return ret;
}

vector<T> mul(const vector<T> &a, const vector<T> &b) {
  assert(a.size() == b.size());
  vector<T> ret(a.size());
  for (int i = 0; i < (int)a.size(); i++) {
    ret[i] = a[i] * b[i];
  }
  return ret;
}

vector<T> divide(const vector<T> &a, const T &b) {
  vector<T> ret(a.size());
  for (int i = 0; i < (int)a.size(); i++) {
    ret[i] = a[i] / b;
  }
  return ret;
}

vector<T> normalize(const vector<T> &a, const T &mean, const T &std) {
  vector<T> ret(a.size());
  for (int i = 0; i < (int)a.size(); i++) {
    ret[i] = (a[i] - mean) / std;
  }
  return ret;
}

// === read data from json files ===

pair<vector<vector<T>>, vector<int>> get_data() {
  ifstream mnist_file("data/mnist_handwritten_test.json");
  nlohmann::json mnist_json;
  mnist_file >> mnist_json;

  vector<vector<T>> images;
  vector<int> labels;
  for (int i = 0; i < (int)mnist_json.size(); i++) {
    images.push_back(
        normalize(divide(mnist_json[i]["image"].get<vector<T>>(), 255.0),
                  0.1307, 0.3081));
    labels.push_back(mnist_json[i]["label"].get<int>());
  }

  return {images, labels};
}

// === tools ===

int get_max_idx(vector<T> v) {
  int max_idx = 0;
  for (int i = 1; i < (int)v.size(); i++) {
    if (v[i] > v[max_idx])
      max_idx = i;
  }
  return max_idx;
}

void print(vector<T> x) {
  for (int i = 0; i < 10; i++) {
    cout << x[i] << ' ';
  }
  cout << endl;
}
