#pragma once
#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/gpu_raptor_translator.h"
#include "nigiri/routing/gpu_types.h"
#include "nigiri/routing/search.h"
#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;
using host_sys_days_test = std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<int, std::ratio<86400>>>;
using gpu_sys_days_test = cuda::std::chrono::time_point<cuda::std::chrono::duration<int, cuda::std::ratio<86400>>>;
template<typename T>
struct interval_test {
  T start;
  T end;
};

template<typename T>
struct gpu_interval_test {
  T start;
  T end;
};
TEST(routing, gpu_raptor) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
  /*auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size()
      );
  *///EXPECT_NE(nullptr, gtt);
  //destroy_gpu_timetable(gtt);
  direction const SearchDir = nigiri::direction::kForward;
  auto rtt = rt_timetable{};
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};
  //test ob unsere daten typen genaus funktionieren wie die cista
  //testen ob raptor geht

  auto const results =
      raptor_search(tt, &rtt, "A", "D", sys_days{May / 2 / 2019} +23h,
                    direction::kForward,true);

  // Initialize host-side date_range_ (interval)
  interval_test<host_sys_days_test> host_date_range;// Convert to the correct duration type (days)
  host_date_range.start = host_sys_days_test(std::chrono::duration<int, std::ratio<86400>>(5)); // Start time 5 days since epoch
  host_date_range.end = host_sys_days_test(std::chrono::duration<int, std::ratio<86400>>(10));  // End time 10 days since epoch

  // Apply reinterpret_cast to convert to gpu_interval<gpu_sys_days>
  gpu_interval_test<gpu_sys_days>* gpu_date_range = reinterpret_cast<gpu_interval_test<gpu_sys_days>*>(&host_date_range);

  // Check that values are correctly interpreted on GPU side
  assert(gpu_date_range->start.time_since_epoch().count() == host_date_range.start.time_since_epoch().count());
  assert(gpu_date_range->end.time_since_epoch().count() == host_date_range.end.time_since_epoch().count());

  // Output results for verification
  std::cout << "Host start days: " << host_date_range.start.time_since_epoch().count() << std::endl;
  std::cout << "Host end days: " << host_date_range.end.time_since_epoch().count() << std::endl;
  std::cout << "GPU start days: " << gpu_date_range->start.time_since_epoch().count() << std::endl;
  std::cout << "GPU end days: " << gpu_date_range->end.time_since_epoch().count() << std::endl;
  std::cout << "Test passed: Host and GPU intervals have matching values." << std::endl;

  auto gpu_direction2 = *reinterpret_cast<const gpu_direction*>(&SearchDir);
  bool tester = gpu_direction2 == gpu_direction::kForward;
  std::cout<<"reinterpret_cast SearchDir is working: " << tester;
  std::cout<<"" << results.size();
}
