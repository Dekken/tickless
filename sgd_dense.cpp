
#include <cmath>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <malloc.h>

#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE ulong
#else
#define INDICE_TYPE std::uint32_t
#endif

#include "cereal/archives/portable_binary.hpp"
#include "array.hpp"
#include "hpp.hpp"

#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

constexpr size_t N_ITER = 11;

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::string labels_s("labels.cereal"), features_s("features.cereal");

  std::vector<size_t> info;
  std::vector<double> data;
  {
    std::ifstream bin_data(features_s, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    tick::load_array2d_with_raw_data(iarchive, data, info);
  }
  tick::Array2DRaw<double> features(data.data(), info.data());

  const size_t N_FEATURES = features.cols(), N_SAMPLES = features.rows();
  std::vector<double> vlabels(N_SAMPLES), iterate(N_FEATURES);
  {
    std::ifstream bin_data(labels_s, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    tick::load_array_with_raw_data(iarchive, vlabels.data());
  }

  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };

  const auto STRENGTH = 1e-2;

  auto call_single = [&](ulong i, const double *coeffs, double step, double *out) {
    if (coeffs[i] < 0) out[i] = 0;
    out[i] = coeffs[i] / (1 + step * STRENGTH);
  };
  auto call = [&](const double *coeffs, double step, double *out, size_t size) {
    for (size_t i = 0; i < size; i++) call_single(i, coeffs, step, out);
  };

  std::vector<double> objs;
  auto start = NOW;
  size_t t = 0;
  for (size_t j = 0; j < N_ITER; ++j) {
    tick::sgd::dense::solve(features, vlabels.data(), iterate.data(), call, next_i, t);
    if (j % 10 == 0)
      objs.emplace_back(tick::logreg::loss(features, vlabels.data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
