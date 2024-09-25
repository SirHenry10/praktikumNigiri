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
void merge_file(const std::string& output_file, int num_parts) {
  std::ofstream outfile(output_file, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Fehler beim Erstellen der Ausgabedatei: " << output_file << std::endl;
    return;
  }

  char buffer[1024];  // Ein kleiner Puffer, um die Teile zu lesen
  for (int part = 0; part < num_parts; ++part) {
    std::string part_file = output_file + ".part" + std::to_string(part);
    std::ifstream infile(part_file, std::ios::binary);

    if (!infile.is_open()) {
      std::cerr << "Fehler beim Öffnen des Teils: " << part_file << std::endl;
      return;
    }

    // Lese den Teil und füge ihn in die Ausgangsdatei ein
    while (!infile.eof()) {
      infile.read(buffer, sizeof(buffer));
      outfile.write(buffer, infile.gcount());
    }

    infile.close();
  }

  outfile.close();
  std::cout << "Datei erfolgreich zusammengefügt: " << output_file << std::endl;
}
std::filesystem::path project_root = std::filesystem::current_path().parent_path();
std::filesystem::path test_path_germany(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs"); //Alle nicht ASCII zeichen entfernt.
fs_dir test_files_germany(test_path_germany);

TEST(routing, gpu_raptor_germany) {
  std::filesystem::path filePath = "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs/stop_times.txt";
  if (!std::filesystem::exists(filePath)) {
    merge_file("test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs/stop_times.txt",5);
  }
  timetable tt;
  std::cout << "Lade Fahrplan..." << std::endl;
  tt.date_range_ = {date::sys_days{2024_y / September / 25},
                    date::sys_days{2024_y / September / 29}}; //test_files_germany only available until December 14
  load_timetable({}, source_idx_t{0}, test_files_germany, tt); //problem glaube mit deutschen sonder zeichen
  std::cout << "Fahrplan geladen." << std::endl;

  std::cout << "Finalisiere Fahrplan..." << std::endl;
  finalize(tt);
  std::cout << "Fahrplan finalisiert." << std::endl;

  std::cout << "Starte Raptor-Suche..." << std::endl;
  //Flensburg Holzkrugweg -> Oberstdorf, Campingplatz
  auto const results_cpu = raptor_search(tt, nullptr, "de:01001:27334::1", "de:09780:9256:0:1",
                                         sys_days{September / 28 / 2024} + 2h,
                                         nigiri::direction::kBackward);

  std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
  auto const results_gpu = raptor_search(tt, nullptr, "de:01001:27334::1", "de:09780:9256:0:1",
                                         sys_days{September / 28 / 2024} + 2h,
                                         nigiri::direction::kBackward,useGPU);

  std::cout << "Raptor-Suche abgeschlossen." << std::endl;
  std::stringstream ss;
  for (auto const& x : results_cpu) {
    x.print(ss, tt);
    ss << "\n";
  }
  std::cout << ss.str() << "\n";
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

constexpr auto const fwd_journeys2 = R"(
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


  auto const results_cpu = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},nonGPU);
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    std::cout << "result cpu\n";
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  auto const results_gpu = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},useGPU);
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    std::cout << "result gpu\n";
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  EXPECT_EQ(std::string_view{fwd_journeys2}, ss2.str());

}