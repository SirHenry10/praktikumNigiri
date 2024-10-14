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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
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
      x.print(ss1, tt);
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
      x.print(ss2, tt);
      ss2 << "\n\n";
    }
    std::cout << ss2.str();
    // Output the benchmarking results
    std::cout << "Stettenhofen -> Feucht";
    std::cout << "CPU Time: " << cpu_duration << " microseconds\n";
    std::cout << "GPU Time: " << gpu_duration << " microseconds\n";
    printf("GPU_size: %d, CPU_size: %d",results_gpu.size(),results_cpu.size());
    EXPECT_EQ(ss1.str(), ss2.str());
  }
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

  constexpr int num_queries = 500;
  std::vector<long long> cpu_times;
  std::vector<long long> gpu_times;
  int matched_queries = 0;

  std::random_device rd;
  unsigned int seed = rd();
  std::cout << "Verwendeter Seed: " << seed << std::endl;

  std::mt19937 gen(seed);
  auto locations = get_locations(tt);

  for (int i = 0; i < num_queries; ++i) {
    auto [start, end] = get_random_location_pair(locations,gen);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr,
                                           start, end,
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss_cpu, ss_gpu;
    ss_cpu << "\n";
    cpu_times.push_back(cpu_duration);
    for (auto const& x : results_cpu) {
      x.print(ss_cpu, tt);
      ss_cpu << "\n\n";
    }

    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           start, end,
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();
    gpu_times.push_back(gpu_duration);

    ss_gpu << "\n";
    for (auto const& x : results_gpu) {
      x.print(ss_gpu, tt);
      ss_gpu << "\n\n";
    }

    EXPECT_EQ(ss_cpu.str(), ss_gpu.str())
        << "Results differ for query " << i + 1 << ": " << start << " -> " << end;

    if (ss_cpu.str() == ss_gpu.str()) {
      matched_queries++;
    }

    if ((i + 1) % 10 == 0) {
      std::cout << "Bearbeitet: " << (i + 1) << " von " << num_queries << " Querys " << std::endl;
    }
  }

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