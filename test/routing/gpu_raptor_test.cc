#pragma once
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
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
using namespace nigiri::loader::gtfs;
using nigiri::test::raptor_search;

namespace {

// T_RE1
// A | 01.05.  49:00 = 03.05. 01:00
// B | 01.05.  50:00 = 03.05. 02:00
//
// T_RE2
// B | 03.05.  00:30
// C | 03.05.  00:45
// D | 03.05.  01:00
//
// => delay T_RE2 at B  [03.05. 00:30]+2h = [03.05. 02:30]
// => search connection from A --> D (only possible if the transfer at B works!)
mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
S_RE1,20190501,1
S_RE2,20190503,1
)"}},
       {path{kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,RE 1,,,3
R_RE2,DB,RE 2,,,3
)"}},
       {path{kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE2,S_RE2,T_RE2,RE 2,
)"}},
       {path{kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,49:00:00,49:00:00,A,1,0,0
T_RE1,50:00:00,50:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,01:00:00,01:00:00,D,3,0,0
)"}}}};
}

constexpr auto const fwd_journeys = R"([2019-05-02 23:00, 2019-05-03 01:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-02 23:00]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (A, A) [2019-05-02 23:00] -> (B, B) [2019-05-03 00:00]
   0: A       A...............................................                               d: 02.05 23:00 [03.05 01:00]  [{name=Bus RE 1, day=2019-05-02, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 00:00 [03.05 02:00]
leg 1: (B, B) [2019-05-03 00:00] -> (B, B) [2019-05-03 00:02]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 02.05 22:30 [03.05 00:30]  RT 03.05 00:30 [03.05 02:30]  [{name=Bus RE 2, day=2019-05-02, id=T_RE2, src=0}]
   1: C       C............................................... a: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  d: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  [{name=Bus RE 2, day=2019-05-02, id=T_RE2, src=0}]
   2: D       D............................................... a: 02.05 23:00 [03.05 01:00]  RT 03.05 01:00 [03.05 03:00]
leg 3: (D, D) [2019-05-03 01:00] -> (D, D) [2019-05-03 01:00]
  FOOTPATH (duration=0)

)";

}  // namespace
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
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};


   std::cerr << "raptor_search_started" << std::endl;
    auto const cpu_results =
        raptor_search(tt, nullptr, "A", "D", sys_days{May / 2 / 2019} +23h,nonGPU);
  std::cerr << "raptor_search_started" << std::endl;
  auto const gpu_results =
      raptor_search(tt, nullptr, "A", "D", sys_days{May / 2 / 2019} +23h,useGPU);

  std::cout<<"" << cpu_results.size();
  std::cout<<"" << gpu_results.size();
}
