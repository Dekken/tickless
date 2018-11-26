
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE ulong
#else
#define INDICE_TYPE std::uint32_t
#endif

namespace tick {

template <class T>
T pow(const T &f, const long double &e = 2) {
  return ::pow(f, e);
}

template <class T>
T abs(const T &f) {
  return f < 0 ? f * -1 : f;
}

template <class T>
T sigmoid(const T z) {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <class T>
T logistic(const T z) {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}

namespace logreg {

template <typename T>
T *get_features_raw(T *features, size_t cols, size_t i) {
  return &features[cols * i];
}

template <typename T>
T dot(const T *t1, const T *t2, size_t size) {
  T res {0};
  for (size_t i = 0; i < size; ++i) res += t1[i] * t2[i];
  return res;
}

template <typename T>
T get_inner_prod(T *features, size_t cols, size_t i, size_t coeffs_size, T *coeffs) {
  return dot(coeffs, get_features_raw(features, cols, i), coeffs_size);
}

// template <typename T>
// T loss_i(T *features, size_t cols, ulong i, T y_i, size_t coeffs_size, T *coeffs) {
//   return logistic(get_inner_prod(features, cols, i, coeffs) * y_i);
// }

template <typename T, typename Sparse2D>
T loss(const Sparse2D &features, T *labels, T *coeffs) {
  T t{0};
  const size_t &cols = features.cols();
  const size_t &rows = features.rows();
  for (size_t i = 0; i < rows; i++)
    t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}

template <typename T>
T grad_i_factor(T *features, size_t cols, ulong i, T y_i, size_t coeffs_size, T *coeffs) {
  return y_i * (sigmoid(y_i * get_inner_prod(features, cols, i, coeffs_size, coeffs)) - 1);
}

template <typename T>
T get_inner_prod(const size_t i, const size_t cols, const size_t rows, T *features, T *coeffs) {
  return dot(coeffs, &features[i * cols], cols);
}
template <typename T>
T grad_i_factor(const size_t i, const size_t cols, const size_t rows, T *features, T *labels,
                T *coeffs) {
  const T y_i = labels[i];
  return y_i * (sigmoid(y_i * get_inner_prod(i, cols, rows, features, coeffs)) - 1);
}

}  // namespace logreg

namespace saga {
using INDEX_TYPE = INDICE_TYPE;
namespace dense {
template <typename T, typename NEXT_I>
void solve(T *features, T *labels, T *gradients_average, T *gradients_memory, T *iterate,
           NEXT_I _next_i) {
  size_t N_FEATURES = 200;
  size_t N_SAMPLES = 75000;
  T N_SAMPLES_inverse = ((double)1 / (double)N_SAMPLES);
  double step = 0.00257480411965l;
  ulong n_features = N_FEATURES;
  for (ulong t = 0; t < N_SAMPLES; ++t) {
    INDEX_TYPE i = _next_i();
    T grad_i_factor = logreg::grad_i_factor(i, N_FEATURES, N_SAMPLES, features, labels, iterate);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    const T *x_i = logreg::get_features_raw(features, N_FEATURES, i);
    for (ulong j = 0; j < n_features; ++j) {
      T grad_avg_j = gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_i[j] + grad_avg_j);
      gradients_average[j] += grad_factor_diff * x_i[j] * N_SAMPLES_inverse;
    }
  }
}
}  // namespace dense

namespace sparse {
template <typename T, typename Sparse2D, typename PROX, typename NEXT_I>
void solve(const Sparse2D &features, T *labels, T *gradients_average, T *gradients_memory,
           T *iterate, T *steps_correction, PROX call_single, NEXT_I _next_i) {
  size_t n_samples = features.rows();
  T n_samples_inverse = ((double)1 / (double)n_samples);
  double step = 0.00257480411965l;
  for (ulong t = 0; t < n_samples; ++t) {
    INDEX_TYPE i = _next_i();
    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDEX_TYPE *x_i_indices = features.row_indices(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      const INDEX_TYPE &j = x_i_indices[idx_nnz];
      iterate[j] -=
          step * (grad_factor_diff * x_i[idx_nnz] + steps_correction[j] * gradients_average[j]);
      gradients_average[j] += grad_factor_diff * x_i[idx_nnz] * n_samples_inverse;
      call_single(j, iterate, step * steps_correction[j], iterate);
    }
  }
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_columns_sparsity(const Sparse2D &features) {
  std::vector<T> column_sparsity(features.cols());
  std::fill(column_sparsity.begin(), column_sparsity.end(), 0);
  double samples_inverse = 1. / features.rows();
  for (ulong i = 0; i < features.rows(); ++i) {
    auto row = features.row_indices(i);
    for (ulong j = 0; j < features.row_size(i); ++j) column_sparsity[row[j]] += 1;
  }
  for (uint64_t i = 0; i < features.cols(); ++i) column_sparsity[i] *= samples_inverse;
  return column_sparsity;
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_step_corrections(const Sparse2D &features) {
  std::vector<T> steps_correction(features.cols()),
      columns_sparsity(compute_columns_sparsity(features));
  for (ulong j = 0; j < features.cols(); ++j) steps_correction[j] = 1. / columns_sparsity[j];
  return steps_correction;
}

}  // namespace sparse
}  // namespace saga

template <class Archive, class T>
void load_array_with_raw_data(Archive &ar, T *data) {
  bool is_sparse = false;
  ar(CEREAL_NVP(is_sparse));
  // if (is_sparse) TICK_ERROR("Deserializing sparse arrays is not supported yet.");
  ulong vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));
  ar(cereal::binary_data(data, static_cast<std::size_t>(vectorSize) * sizeof(T)));
}

template <class Archive, class T>
void load_sparse2d_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                 std::vector<INDICE_TYPE> &indices,
                                 std::vector<INDICE_TYPE> &row_indices) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse, rows, cols, size);

  data.resize(data.size() + size_sparse);
  info.resize(info.size() + 5);
  indices.resize(indices.size() + size_sparse);
  row_indices.resize(row_indices.size() + rows + 1);

  T *s_data = &data[data.size()] - size_sparse;
  size_t *s_info = &info[info.size()] - 5;
  INDICE_TYPE *s_indices = &indices[indices.size()] - size_sparse;
  INDICE_TYPE *s_row_indices = &row_indices[row_indices.size()] - (rows + 1);

  ar(cereal::binary_data(s_data, sizeof(T) * size_sparse));
  ar(cereal::binary_data(s_indices, sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(s_row_indices, sizeof(INDICE_TYPE) * (rows + 1)));

  s_info[0] = cols;
  s_info[1] = rows;
  s_info[2] = size_sparse;
  s_info[3] = data.size() - size_sparse;
  s_info[4] = row_indices.size() - (rows + 1);
}

}  // namespace tick
