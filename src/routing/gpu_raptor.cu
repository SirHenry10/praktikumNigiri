#include "nigiri/routing/gpu_raptor.cuh"

extern "C" {


static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
static auto get_best(auto x, auto... y) {
  ((x = get_best(x, y)), ...);
  return x;
}
static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

void reset_arrivals(){

}

void next_start_time(){

}

void add_start()
}  // extern "C"
