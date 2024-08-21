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
  auto testeras = *reinterpret_cast<gpu_vecvec<gpu_location_idx_t, gpu_route_idx_t>*>(&tt.location_routes_);
  auto test12_gpu = testeras.bucket_starts_[1];
  auto test12_cpu = tt.location_routes_.bucket_starts_[1];
  ASSERT_EQ(test12_cpu,test12_gpu);
  //testen ob raptor geht
  /*
  auto const results =
      raptor_search(tt, &rtt, "A", "D", sys_days{May / 2 / 2019} +23h,
                    nigiri::direction::kBackward,true); //TODO: warum funktioniert nicht ohne direction?? siehe rt_test
  */
   //auto gpu_direction2 = *reinterpret_cast<enum gpu_direction*>(&SearchDir);
  //bool tester = gpu_direction2 == gpu_direction::kForward;
  //std::cout<<"" << results.size();
  gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* te = reinterpret_cast<gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>*>(&tt.location_routes_);
  std::cout<<"erster Wert: " << te->bucket_starts_ <<" \n"<< tt.location_routes_.bucket_starts_ << "\n";
  std::cout<<"";
}
