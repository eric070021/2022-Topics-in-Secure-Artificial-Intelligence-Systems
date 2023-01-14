#ifndef UTILS_H
#define UTILS_H
#include "params.h"

void load_matrix(const string npy_file, vector<vector<T>> &matrix);
void load_array(const string npy_file, vector<T> &arr);
void load_linear(const string linear_name, vector<vector<T>> &weight,
                 vector<T> &bias);
vector<T> matmul(const vector<vector<T>> &M, const vector<T> &v);
vector<T> add(const vector<T> &a, const vector<T> &b);
vector<T> add(const vector<T> &a, const T &b);
vector<T> mul(const vector<T> &a, const vector<T> &b);
vector<T> divide(const vector<T> &a, const T &b);
vector<T> normalize(const vector<T> &a, const T &mean, const T &std);
pair<vector<vector<T>>, vector<int>> get_data();
int get_max_idx(vector<T>);
void print(vector<T> x);

#endif
