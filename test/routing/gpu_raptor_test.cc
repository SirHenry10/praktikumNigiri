#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/gpu_timetable.h"

#include "nigiri/routing/gpu_types.h"
#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

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

  cista::raw::gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* te = reinterpret_cast<cista::raw::gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t>*>(&tt.location_routes_);
  std::cout<<"erster Wert: " << te->bucket_starts_ <<" \n"<< tt.location_routes_.bucket_starts_ << "\n";
  uint32_t tester2 = 22222222;
  auto tester = gpu_clamp(tester2);
  std::cout<<"";
}
