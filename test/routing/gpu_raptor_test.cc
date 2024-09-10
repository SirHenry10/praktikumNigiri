#pragma once
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/routing/gpu_raptor_translator.h"
#include "nigiri/routing/gpu_types.h"
#include "nigiri/routing/ontrip_train.h"
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

TEST(routing, gpu_timetable) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);
  EXPECT_NE(nullptr, gtt);
  destroy_gpu_timetable(gtt);
  EXPECT_EQ(nullptr, gtt);
}
TEST(routing, gpu_types) {
  auto SearchDir_kForward = direction::kForward;
  auto gpu_direction_kForward = *reinterpret_cast<const gpu_direction*>(&SearchDir_kForward);
  auto SearchDir_kBackward = direction::kBackward;
  auto gpu_direction_kBackward = *reinterpret_cast<const gpu_direction*>(&SearchDir_kBackward);
  static gpu_direction const gpu_direction_static_kForward =
      static_cast<enum gpu_direction const>(SearchDir_kForward);
  static gpu_direction const gpu_direction_static_kBackward =
      static_cast<enum gpu_direction const>(SearchDir_kBackward);
  EXPECT_EQ(gpu_direction_static_kForward, gpu_direction::kForward);
  EXPECT_EQ(gpu_direction_static_kBackward, gpu_direction::kBackward);
  EXPECT_EQ(gpu_direction_kForward, gpu_direction::kForward);
  EXPECT_EQ(gpu_direction_kBackward, gpu_direction::kBackward);
  //TODO: mehr typen noch test...
}
TEST(routing, gpu_raptor) {
  using namespace date;
  timetable tt;
  tt.date_range_ = full_period();
  constexpr auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);

  auto const t = get_ref_transport(
      tt, {"3374/0000008/1350/0000006/2950/", source_idx_t{0}},
      March / 29 / 2020, false);
  ASSERT_TRUE(t.has_value());

  auto q = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};
  generate_ontrip_train_query(tt, t->first, 1, q);

  auto const cpu_results = raptor_search(tt, nullptr, std::move(q),nonGPU);
  auto const gpu_results = raptor_search(tt, nullptr, std::move(q),useGPU);

  std::stringstream ss;
  ss << "\n";
  for (auto const& x :  gpu_results) {
    std::cout << "result gpu\n";
    x.print(std::cout, tt);
    ss << "\n\n";
  }
  std::cout << "results cpu: " <<  cpu_results.size() << "\n";
  std::cout << "results gpu: " <<  gpu_results.size() << "\n";
  ASSERT_EQ(cpu_results.size(),gpu_results.size());
}

constexpr auto const fwd_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:15]
  FOOTPATH (duration=0)


[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 3: (C, 0000003) [2020-03-30 07:45] -> (C, 0000003) [2020-03-30 07:45]
  FOOTPATH (duration=0)


)";

TEST(routing, gpu_raptor_forward) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  auto const results = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},useGPU);
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  EXPECT_EQ(std::string_view{fwd_journeys}, ss.str());
}