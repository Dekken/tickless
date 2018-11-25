

#include <iostream>
#include <fstream>
#include <malloc.h>
#include <math.h>
#include <vector>
#include <random>
#include <cmath>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  constexpr size_t A = 256;
  size_t I = 4;

  size_t *a = (size_t *)aligned_alloc(A, I * sizeof(size_t));
  size_t *b = (size_t *)aligned_alloc(A, I * sizeof(size_t));
  size_t *c = (size_t *)aligned_alloc(A, I * sizeof(size_t));
  size_t *d = (size_t *)aligned_alloc(A, I * sizeof(size_t));

  for (size_t i = 0; i != I; ++i) {
    a[i] = b[i] = c[i] = d[i] = i;
  }

  for (size_t i = 0; i < I; i++) a[i] = std::fma(b[i], c[i], d[i]);
  for (size_t i = 0; i < I; i++) a[i] = (b[i] * c[i]) + d[i];
  for (size_t i = 0; i < I; i++) std::cout << "a[i] " << a[i] << std::endl;

  free(a);
  free(b);
  free(c);
  free(d);

  return 0;
};
