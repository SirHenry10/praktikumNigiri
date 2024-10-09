#pragma once
#include <random>
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
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
std::filesystem::path test_path_germany_zip(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs.zip");
std::filesystem::path test_path_germany(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs");
auto const german_dir_zip = zip_dir{test_path_germany_zip};
auto const german_dir = fs_dir{test_path_germany};
TEST(routing, gpu_raptor_germany) {
  timetable tt;
  std::cout << "Lade Fahrplan..." << std::endl;
  // muss hier full_period hin?
  tt.date_range_ = {date::sys_days{2024_y / September / 25},
                    date::sys_days{2024_y / September / 26}}; //test_files_germany only available until December 14
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, german_dir_zip, tt);
  std::cout << "Fahrplan geladen." << std::endl;

  std::cout << "Finalisiere Fahrplan..." << std::endl;
  loader::finalize(tt);
  std::cout << "Fahrplan finalisiert." << std::endl;
  auto gtt = translate_tt_in_gtt(tt);
  {
    // 000000001151 -> de:11000:900152005::3
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "000300184068", "000000012482",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "000300184068", "000000012482",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "000300184068 -> 000000012482";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  /*
  {
    // Frankfurt Hauptwache - de:06412:1:21:3 -> Darmstadt Nordbf - de:06411:4720
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:06412:1:21:3", "de:06411:4720",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:06412:1:21:3", "de:06411:4720",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Frankfurt Hauptwache -> Darmstadt Nordbf";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // München Hbf - de:09162:100_G -> S+U Berlin Hbf - de:11000:900003200
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:09162:100_G", "de:11000:900003200",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:09162:100_G", "de:11000:900003200",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "München Hbf -> Berlin Hbf";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Köln Neumarkt - de:05315:11111:4:16 -> Köln Ehrenfeld Bf - de:05315:14201
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:05315:11111:4:16", "de:05315:14201",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:05315:11111:4:16", "de:05315:14201",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Köln Neumarkt -> Köln Ehrenfeld";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Heidelberg Rohrbach de:08221:1203 -> Recklinhausen Hbf de:05562:3581
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:08221:1203", "de:05562:3581",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:08221:1203", "de:05562:3581",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "HD Rohrbach -> Recklinghausen Hbf";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Schaafheim - de:06432:22576:1:1 -> Reinheim - de:06432:24562:1:1
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:06432:22576:1:1", "de:06432:24562:1:1",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:06432:22576:1:1", "de:06432:24562:1:1",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Schaafheim -> Reinheim";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Dresden Hbf - de:14612:36 -> Hamburg Hbf - de:02000:10002
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:14612:36", "de:02000:10002",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:14612:36", "de:02000:10002",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Dresden Hbf -> Hamburg Hbf";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Ludwigshafen Mitte - de:07314:2019 -> Mannheim Wasserturm de:08222:2475:2:5
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:07314:2019", "de:08222:2475:2:5",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:07314:2019", "de:08222:2475:2:5",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "LU Mitte -> Mannheim Wasserturm";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Böblingen Südbf - de:08115:6742 -> Zuffenhausen Rathaus - de:08111:108:80
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:08115:6742", "de:08111:108:80",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:08115:6742", "de:08111:108:80",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Böblingen -> Zuffenhausen";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Lübeck Drägerwerk - de:01003:57748::1 -> Lennestadt - de:05966:10238_G
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:01003:57748::1", "de:05966:10238_G",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:01003:57748::1", "de:05966:10238_G",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Lübeck -> Lennestadt";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
  {
    // Stettenhofen - de:09772:4204:0:A -> Feucht Ost - de:09574:7210
    std::cout << "Starte Raptor-Suche..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    auto const results_cpu = raptor_search(tt, nullptr,
                                           "de:09772:4204:0:A", "de:09574:7210",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x :  results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    std::cout << ss1.str();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Starte GPU-Raptor-Suche..." << std::endl;
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           "de:09772:4204:0:A", "de:09574:7210",
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    std::cout << "Raptor-Suche abgeschlossen." << std::endl;

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x :  results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Stettenhofen -> Feucht";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }*/
  destroy_gpu_timetable(gtt);
}

TEST(routing, gpu_raptor_on_train_1) {
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
  auto gtt = translate_tt_in_gtt(tt);
  auto q1 = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};

  generate_ontrip_train_query(tt, t->first, 1, q1);
  auto const results_cpu = raptor_search(tt, nullptr, std::move(q1));
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  auto q2 = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};
  generate_ontrip_train_query(tt, t->first, 1, q2);
  auto const results_gpu = raptor_search(tt, nullptr,gtt, std::move(q2));
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  std::cout << ss2.str();
  EXPECT_EQ(ss1.str(), ss2.str());
  destroy_gpu_timetable(gtt);
}




TEST(routing, gpu_raptor_forward) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);
  {
    // CPU Benchmarking
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(
        tt, nullptr, "0000002", "0000003",
        interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                 unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_cpu - start_cpu)
                            .count();

    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x : results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    // GPU Benchmarking
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto const results_gpu = raptor_search(
        tt, nullptr, gtt, "0000002", "0000003",
        interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                 unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gpu - start_gpu)
                            .count();

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x : results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    EXPECT_EQ(ss1.str(), ss2.str());

    // Output the benchmarking results
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
  }
  {
    // CPU Benchmarking
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(
        tt, nullptr, "0000002", "0000003",
        interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                 unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_cpu - start_cpu)
                            .count();

    std::stringstream ss1;
    ss1 << "\n";
    for (auto const& x : results_cpu) {
      x.print(std::cout, tt);
      ss1 << "\n\n";
    }
    // GPU Benchmarking
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto const results_gpu = raptor_search(
        tt, nullptr, gtt, "0000002", "0000003",
        interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                 unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gpu - start_gpu)
                            .count();

    std::stringstream ss2;
    ss2 << "\n";
    for (auto const& x : results_gpu) {
      x.print(std::cout, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    EXPECT_EQ(ss1.str(), ss2.str());

    // Output the benchmarking results
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
  }
  destroy_gpu_timetable(gtt);
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

  auto q1 = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};

  generate_ontrip_train_query(tt, t->first, 1, q1);

  auto const results_cpu = raptor_search(tt, nullptr, std::move(q1));
  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  auto q2 = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};
  generate_ontrip_train_query(tt, t->first, 1, q2);
  auto const results_gpu = raptor_search(tt, nullptr,gtt, std::move(q2));
  printf("GPU:");
  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x :  results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  std::cout << ss2.str();
  EXPECT_EQ(ss1.str(), ss2.str());
  destroy_gpu_timetable(gtt);
}

namespace {
mem_dir gtrfs_test_files() {
  return mem_dir::read(R"(
     "(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_fare_url,agency_id
"grt",https://grt.ca,America/New_York,en,519-585-7555,http://www.grt.ca/en/fares/FarePrices.asp,grt

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,wheelchair_boarding,platform_code
2351,2351,Block Line Station,,  43.422095, -80.462740,,
1033,1033,Block Line / Hanover,,  43.419023, -80.466600,,,0,,1,
2086,2086,Block Line / Kingswood,,  43.417796, -80.473666,,,0,,1,
2885,2885,Block Line / Strasburg,,  43.415733, -80.480340,,,0,,1,
2888,2888,Block Line / Laurentian,,  43.412766, -80.491494,,,0,,1,
3189,3189,Block Line / Westmount,,  43.411515, -80.498966,,,0,,1,
3895,3895,Fischer-Hallman / Westmount,,  43.406717, -80.500091,,,0,,1,
3893,3893,Fischer-Hallman / Activa,,  43.414221, -80.508534,,,0,,1,
2969,2969,Fischer-Hallman / Ottawa,,  43.416570, -80.510880,,,0,,1,
2971,2971,Fischer-Hallman / Mcgarry,,  43.423420, -80.518818,,,0,,1,
2986,2986,Fischer-Hallman / Queens,,  43.428585, -80.523337,,,0,,1,
3891,3891,Fischer-Hallman / Highland,,  43.431587, -80.525376,,,0,,1,
3143,3143,Fischer-Hallman / Victoria,,  43.436843, -80.529202,,,0,,1,
3144,3144,Fischer-Hallman / Stoke,,  43.439462, -80.535435,,,0,,1,
3146,3146,Fischer-Hallman / University Ave.,,  43.444402, -80.545691,,,0,,1,
1992,1992,Fischer-Hallman / Thorndale,,  43.448678, -80.550034,,,0,,1,
1972,1972,Fischer-Hallman / Erb,,  43.452906, -80.553686,,,0,,1,
3465,3465,Fischer-Hallman / Keats Way,,  43.458370, -80.557824,,,0,,1,
3890,3890,Fischer-Hallman / Columbia,,  43.467368, -80.565646,,,0,,1,
1117,1117,Columbia / U.W. - Columbia Lake Village,,  43.469091, -80.561788,,,0,,1,
3899,3899,Columbia / University Of Waterloo,,  43.474462, -80.546591,,,0,,1,
1223,1223,University Of Waterloo Station,,  43.474023, -80.540433,,
3887,3887,Phillip / Columbia,,  43.476409, -80.539399,,,0,,1,
2524,2524,Columbia / Hazel,,  43.480027, -80.531130,,,0,,1,
4073,4073,King / Columbia,,  43.482448, -80.526106,,,0,,1,
1916,1916,King / Weber,,  43.484988, -80.526677,,,0,,1,
1918,1918,King / Manulife,,  43.491207, -80.528026,,,0,,1,
1127,1127,Conestoga Station,,  43.498036, -80.528999,,

# calendar_dates.txt
service_id,date,exception_type
201-Weekday-66-23SUMM-1111100,20230703,1
201-Weekday-66-23SUMM-1111100,20230704,1
201-Weekday-66-23SUMM-1111100,20230705,1
201-Weekday-66-23SUMM-1111100,20230706,1
201-Weekday-66-23SUMM-1111100,20230707,1
201-Weekday-66-23SUMM-1111100,20230710,1
201-Weekday-66-23SUMM-1111100,20230711,1
201-Weekday-66-23SUMM-1111100,20230712,1
201-Weekday-66-23SUMM-1111100,20230713,1
201-Weekday-66-23SUMM-1111100,20230714,1
201-Weekday-66-23SUMM-1111100,20230717,1
201-Weekday-66-23SUMM-1111100,20230718,1
201-Weekday-66-23SUMM-1111100,20230719,1
201-Weekday-66-23SUMM-1111100,20230720,1
201-Weekday-66-23SUMM-1111100,20230721,1
201-Weekday-66-23SUMM-1111100,20230724,1
201-Weekday-66-23SUMM-1111100,20230725,1
201-Weekday-66-23SUMM-1111100,20230726,1
201-Weekday-66-23SUMM-1111100,20230727,1
201-Weekday-66-23SUMM-1111100,20230728,1
201-Weekday-66-23SUMM-1111100,20230731,1
201-Weekday-66-23SUMM-1111100,20230801,1
201-Weekday-66-23SUMM-1111100,20230802,1
201-Weekday-66-23SUMM-1111100,20230803,1
201-Weekday-66-23SUMM-1111100,20230804,1
201-Weekday-66-23SUMM-1111100,20230808,1
201-Weekday-66-23SUMM-1111100,20230809,1
201-Weekday-66-23SUMM-1111100,20230810,1
201-Weekday-66-23SUMM-1111100,20230811,1
201-Weekday-66-23SUMM-1111100,20230814,1
201-Weekday-66-23SUMM-1111100,20230815,1
201-Weekday-66-23SUMM-1111100,20230816,1
201-Weekday-66-23SUMM-1111100,20230817,1
201-Weekday-66-23SUMM-1111100,20230818,1
201-Weekday-66-23SUMM-1111100,20230821,1
201-Weekday-66-23SUMM-1111100,20230822,1
201-Weekday-66-23SUMM-1111100,20230823,1
201-Weekday-66-23SUMM-1111100,20230824,1
201-Weekday-66-23SUMM-1111100,20230825,1
201-Weekday-66-23SUMM-1111100,20230828,1
201-Weekday-66-23SUMM-1111100,20230829,1
201-Weekday-66-23SUMM-1111100,20230830,1
201-Weekday-66-23SUMM-1111100,20230831,1
201-Weekday-66-23SUMM-1111100,20230901,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
201,grt,iXpress Fischer-Hallman,,3,https://www.grt.ca/en/schedules-maps/schedules.aspx

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
201,201-Weekday-66-23SUMM-1111100,3248651,Conestoga Station,0,340341,2010025,1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
3248651,05:15:00,05:15:00,2351,1,0,0
3248651,05:16:00,05:16:00,1033,2,0,0
3248651,05:18:00,05:18:00,2086,3,0,0
3248651,05:19:00,05:19:00,2885,4,0,0
3248651,05:21:00,05:21:00,2888,5,0,0
3248651,05:22:00,05:22:00,3189,6,0,0
3248651,05:24:00,05:24:00,3895,7,0,0
3248651,05:26:00,05:26:00,3893,8,0,0
3248651,05:27:00,05:27:00,2969,9,0,0
3248651,05:29:00,05:29:00,2971,10,0,0
3248651,05:31:00,05:31:00,2986,11,0,0
3248651,05:32:00,05:32:00,3891,12,0,0
3248651,05:33:00,05:33:00,3143,13,0,0
3248651,05:35:00,05:35:00,3144,14,0,0
3248651,05:37:00,05:37:00,3146,15,0,0
3248651,05:38:00,05:38:00,1992,16,0,0
3248651,05:39:00,05:39:00,1972,17,0,0
3248651,05:40:00,05:40:00,3465,18,0,0
3248651,05:42:00,05:42:00,3890,19,0,0
3248651,05:43:00,05:43:00,1117,20,0,0
3248651,05:46:00,05:46:00,3899,21,0,0
3248651,05:47:00,05:49:00,1223,22,0,0
3248651,05:50:00,05:50:00,3887,23,0,0
3248651,05:53:00,05:53:00,2524,24,0,0
3248651,05:54:00,05:54:00,4073,25,0,0
3248651,05:55:00,05:55:00,1916,26,0,0
3248651,05:56:00,05:56:00,1918,27,0,0
3248651,05:58:00,05:58:00,1127,28,1,0
)");
}
}  // namespace
TEST(routing, gtfs_gpu_raptor) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, gtrfs_test_files(), tt);
  finalize(tt);
  auto gtt = translate_tt_in_gtt(tt);

  // CPU Benchmarking
  auto start_cpu = std::chrono::high_resolution_clock::now();
  auto const results_cpu = raptor_search(
      tt, nullptr, "2351", "1127",
      interval{unixtime_t{sys_days{2023_y / August / 10}} + 5_hours,
               unixtime_t{sys_days{2023_y / August / 11}} + 6_hours});
  auto end_cpu = std::chrono::high_resolution_clock::now();
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

  std::stringstream ss1;
  ss1 << "\n";
  for (auto const& x :  results_cpu) {
    x.print(std::cout, tt);
    ss1 << "\n\n";
  }
  // GPU Benchmarking
  auto start_gpu = std::chrono::high_resolution_clock::now();
  auto const results_gpu = raptor_search(
      tt, nullptr, gtt, "2351", "1127",
      interval{unixtime_t{sys_days{2023_y / August / 10}} + 5_hours,
               unixtime_t{sys_days{2023_y / August / 11}} + 6_hours});
  auto end_gpu = std::chrono::high_resolution_clock::now();
  auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

  std::stringstream ss2;
  ss2 << "\n";
  for (auto const& x : results_gpu) {
    x.print(std::cout, tt);
    ss2 << "\n\n";
  }
  std::cout << ss2.str();
  //EXPECT_EQ(ss1.str(), ss2.str());

  // Output the benchmarking results
  std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
  std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
  destroy_gpu_timetable(gtt);
}
std::vector<std::basic_string_view<char>> get_locations(const timetable& tt) {
  std::vector<std::basic_string_view<char>> locations;
  locations.reserve(tt.n_locations());

  for (int i = 0; i < tt.n_locations(); ++i) {
    locations.push_back(tt.locations_.get(location_idx_t{i}).id_);
  }

  return locations;
}

double calculate_average(const std::vector<long long>& times) {
  if (times.empty()){
    return 0.0;
  }
  long long total = std::accumulate(times.begin(), times.end(), 0LL);
  return static_cast<double>(total) / times.size();
}
long long calculate_percentile(std::vector<long long>& times,double number) {
  std::sort(times.begin(), times.end());
  size_t idx = static_cast<size_t>(number * times.size());
  return times[idx];
}

std::pair<std::basic_string_view<char>, std::basic_string_view<char>> get_random_location_pair(const std::vector<std::basic_string_view<char>>& locations, std::mt19937& gen) {
  std::uniform_int_distribution<> dis(0, locations.size() - 1);

  std::basic_string_view<char> start_station = locations[dis(gen)];
  std::basic_string_view<char> end_station = locations[dis(gen)];

  return {start_station, end_station};
}

TEST(routing, gpu_benchmark) {
  timetable tt;
  std::cout << "Lade Fahrplan..." << std::endl;
  tt.date_range_ = {date::sys_days{2024_y / September / 25},
                    date::sys_days{2024_y / September / 26}}; //test_files_germany only available until December 14
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, german_dir_zip, tt);
  std::cout << "Fahrplan geladen." << std::endl;

  std::cout << "Finalisiere Fahrplan..." << std::endl;
  loader::finalize(tt);
  std::cout << "Fahrplan finalisiert." << std::endl;
  auto gtt = translate_tt_in_gtt(tt);
  constexpr int num_queries = 30;
  std::vector<long long> cpu_times;
  std::vector<long long> gpu_times;
  int matched_queries = 0;

  std::random_device rd;
  unsigned int seed = 4081258738;
  std::cout << "Verwendeter Seed: " << seed << std::endl;

  std::mt19937 gen(seed);
  auto locations = get_locations(tt);

  for (int i = 0; i < num_queries; ++i) {
    auto [start, end] = get_random_location_pair(locations,gen);

    // CPU-Suche
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr, start, end,  interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                                             unixtime_t{sys_days{2024_y / September / 25}} + 13_hours}, nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    cpu_times.push_back(cpu_duration);

    // GPU-Suche
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto const results_gpu = raptor_search(tt, nullptr, gtt, start, end, interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                                                  unixtime_t{sys_days{2024_y / September / 25}} + 13_hours}, nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();
    gpu_times.push_back(gpu_duration);

    // Ergebnisse vergleichen
    std::stringstream ss_cpu, ss_gpu;
    ss_cpu << "\n";
    ss_gpu << "\n";
    for (auto const& x : results_cpu) {
      x.print(ss_cpu, tt);
      ss_cpu << "\n\n";
    }
    for (auto const& x : results_gpu) {
      x.print(ss_gpu, tt);
      ss_gpu << "\n\n";
    }

    // Verwenden von EXPECT_EQ mit zusätzlicher Ausgabe der Start- und Endstation
    EXPECT_EQ(ss_cpu.str(), ss_gpu.str())
        << "Results differ for query " << i + 1 << ": " << start << " -> " << end;

    if (ss_cpu.str() == ss_gpu.str()) {
      matched_queries++;
    }
    if ((i + 1) % 10 == 0) {
      std::cout << "Bearbeitet: " << (i + 1) << " von " << num_queries << " Querys " << std::endl;
    }
  }

  // Berechnungen am Ende
  double avg_cpu_time = calculate_average(cpu_times);
  double avg_gpu_time = calculate_average(gpu_times);
  long long cpu_99th = calculate_percentile(cpu_times,0.99);
  long long gpu_99th = calculate_percentile(gpu_times,0.99);
  long long cpu_90th = calculate_percentile(cpu_times,0.90);
  long long gpu_90th = calculate_percentile(gpu_times,0.90);
  long long cpu_50th = calculate_percentile(cpu_times,0.50);
  long long gpu_50th = calculate_percentile(gpu_times,0.50);
  long long cpu_10th = calculate_percentile(cpu_times,0.10);
  long long gpu_10th = calculate_percentile(gpu_times,0.10);
  long long cpu_01th = calculate_percentile(cpu_times,0.01);
  long long gpu_01th = calculate_percentile(gpu_times,0.01);

  // Benchmark-Ergebnisse ausgeben
  std::cout << "Average CPU Time: " << avg_cpu_time << " microseconds\n";
  std::cout << "Average GPU Time: " << avg_gpu_time << " microseconds\n";
  std::cout << "99th Percentile CPU Time: " << cpu_99th << " microseconds\n";
  std::cout << "99th Percentile GPU Time: " << gpu_99th << " microseconds\n";
  std::cout << "90th Percentile CPU Time: " << cpu_90th << " microseconds\n";
  std::cout << "90th Percentile GPU Time: " << gpu_90th << " microseconds\n";
  std::cout << "50th Percentile CPU Time: " << cpu_50th << " microseconds\n";
  std::cout << "50th Percentile GPU Time: " << gpu_50th << " microseconds\n";
  std::cout << "10th Percentile CPU Time: " << cpu_10th << " microseconds\n";
  std::cout << "10th Percentile GPU Time: " << gpu_10th << " microseconds\n";
  std::cout << "01th Percentile CPU Time: " << cpu_01th << " microseconds\n";
  std::cout << "01th Percentile GPU Time: " << gpu_01th << " microseconds\n";
  std::cout << "Matched Queries: " << matched_queries << "/" << num_queries << "\n";

}