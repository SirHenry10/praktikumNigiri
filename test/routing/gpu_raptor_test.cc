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
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using namespace std::chrono_literals;
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
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
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
  auto gpu_date_ranges = reinterpret_cast<nigiri::gpu_interval<gpu_sys_days>*>(&tt.date_range_);
  auto val0 = gpu_date_ranges->from_;
  auto val1 = tt.date_range_.from_;
  auto val3 = gpu_date_ranges->to_;
  auto val4 = tt.date_range_.to_;
  printf("val0: %d",val0);
  printf("val1: %d",val1);
  printf("val3: %d",val3);
  printf("val4: %d",val4);
  auto gtt_bitfields = reinterpret_cast<gpu_vector_map<gpu_bitfield_idx_t,gpu_bitfield>*>(
      &tt.bitfields_);
  for (int i = 0; i < tt.bitfields_.size(); ++i) {
    for (int j = 0; j < tt.bitfields_.data()[i].blocks_.size(); ++j) {
      gtt_bitfields->data()[i].blocks_[j] = tt.bitfields_.data()[i].blocks_[j];
    }
  }

  auto gtt_route_stop_time_ranges = reinterpret_cast<gpu_vector_map<gpu_route_idx_t,
                                                                    nigiri::gpu_interval<std::uint32_t>>*>(
      &tt.route_stop_time_ranges_);

}
std::filesystem::path project_root = std::filesystem::current_path().parent_path();
std::filesystem::path test_path_germany(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs"); //Alle nicht ASCII zeichen entfernt.
fs_dir test_files_germany(test_path_germany);

TEST(routing, gpu_raptor_germany) {
  timetable tt;
  std::cout << "Lade Fahrplan..." << std::endl;
  tt.date_range_ = {date::sys_days{2024_y / September / 28},
                    date::sys_days{2024_y / September / 29}}; //test_files_germany only available until December 14
  load_timetable({}, source_idx_t{0}, test_files_germany, tt); //files müssen ohne nicht ascii zeichen sein!
  std::cout << "Fahrplan geladen." << std::endl;

  std::cout << "Finalisiere Fahrplan..." << std::endl;
  finalize(tt);
  std::cout << "Fahrplan finalisiert." << std::endl;
  auto gtt = translate_tt_in_gtt(tt);

  std::cout << "Starte Raptor-Suche..." << std::endl;
  //Flensburg Holzkrugweg -> Oberstdorf, Campingplatz
  auto const results_cpu = raptor_search(tt, nullptr, "de:01001:27334::1", "de:09780:9256:0:1",
                                         sys_days{September / 28 / 2024} + 2h,
                                         nigiri::direction::kBackward);

  std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
  auto const results_gpu = raptor_search(tt, nullptr ,gtt, "de:01001:27334::1", "de:09780:9256:0:1",
                                         sys_days{September / 28 / 2024} + 2h,
                                         nigiri::direction::kBackward);

  std::cout << "Raptor-Suche abgeschlossen." << std::endl;
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  EXPECT_EQ(ss1.str(), ss2.str());
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


  auto gtt = translate_tt_in_gtt(tt);

  generate_ontrip_train_query(tt, t->first, 1, q);
  auto const results_cpu = raptor_search(tt, nullptr, std::move(q));
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  generate_ontrip_train_query(tt, t->first, 1, q);
  auto const results_gpu = raptor_search(tt, nullptr,gtt, std::move(q));
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  EXPECT_EQ(ss1.str(), ss2.str());
}



TEST(routing, gpu_raptor_forward) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);

  auto const results_cpu = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});

  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  auto const results_gpu = raptor_search(
      tt, nullptr,gtt, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});

   std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  EXPECT_EQ(ss1.str(), ss2.str());
}

TEST(routing, gpu_raptor_backwards) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);

  auto const results_cpu = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},direction::kBackward);
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  auto const results_gpu = raptor_search(
      tt, nullptr,gtt, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},direction::kBackward);

  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  EXPECT_EQ(ss1.str(), ss2.str());
  destroy_gpu_timetable(gtt);
}
TEST(routing, gpu_raptor_ontrip_train) {
  using namespace date;
  timetable tt;
  tt.date_range_ = full_period();
  constexpr auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);

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
  /*
  generate_ontrip_train_query(tt, t->first, 1, q);

  auto const results_cpu = raptor_search(tt, nullptr, std::move(q));
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
   */
  generate_ontrip_train_query(tt, t->first, 1, q);
  auto const results_gpu = raptor_search(tt, nullptr,gtt, std::move(q));
  printf("GPU:");
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  //EXPECT_EQ(ss1.str(), ss2.str());
  destroy_gpu_timetable(gtt);
}
