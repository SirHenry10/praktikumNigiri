#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/gpu_timetable.h"

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
  auto gtt = create_gpu_timetable(
      reinterpret_cast<gpu_delta*>(tt.route_stop_times_.data()),
      tt.route_stop_times_.size()
      );
  EXPECT_NE(nullptr, gtt);
  destroy_gpu_timetable(gtt);

  //Test ob umrechnen funktioniert
  gpu_delta test{12,3};

  date::sys_days testDays{2019_y / March / 25};
  auto test2 = gpu_delta_to_unixtime(testDays,test);
  auto test3 = unix_to_gpu_delta(testDays,test2);
  EXPECT_EQ(test.days_, test3.days_);
  EXPECT_EQ(test.mam_, test3.mam_);
}
