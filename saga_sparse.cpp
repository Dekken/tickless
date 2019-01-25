
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

const constexpr size_t N_ITER = 111;

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  std::string labels_s("url.labels.cereal"), features_s("url.features.cereal");

  std::vector<size_t> info;
  std::vector<double> data;
  std::vector<INDICE_TYPE> indices, row_indices;
  {
    std::ifstream bin_data(features_s, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    tick::load_sparse2d_with_raw_data(iarchive, data, info, indices, row_indices);
  }
  tick::Sparse2DList<double> features(data, info, indices, row_indices);

  const size_t N_SAMPLES = features[0].rows();
  const size_t N_FEATURES = features[0].cols();

  std::vector<double> vlabels(N_SAMPLES);
  {
    std::ifstream bin_data(labels_s, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    tick::load_array_with_raw_data(iarchive, vlabels.data());
  }
  std::vector<double> gradients_average(N_FEATURES), gradients_memory(N_SAMPLES),
      iterate(N_FEATURES),
      steps_corrections(tick::saga::sparse::compute_step_corrections(features[0]));
  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const auto BETA = 1e-10;
  const auto STRENGTH = (1. / N_SAMPLES) + BETA;
  auto call_single = [&](ulong i, const double *coeffs, double step, double *out) {
    if (coeffs[i] < 0) out[i] = 0;
    out[i] = coeffs[i] / (1 + step * STRENGTH);
  };
  std::vector<double> objs;
  auto start = NOW;
  size_t t = 0;
  for (size_t j = 0; j < N_ITER; ++j) {
    tick::saga::sparse::solve(features[0], vlabels.data(), gradients_average.data(),
                              gradients_memory.data(), iterate.data(), steps_corrections.data(),
                              call_single, next_i);
    if (j % 10 == 0)
      objs.emplace_back(tick::logreg::loss(features[0], vlabels.data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
