#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../loader/hrd/hrd_timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

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

constexpr auto const fwd_journeys = R"(
)";

TEST(routing, rt_raptor_forward) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt_timetable{};
  rtt.transport_traffic_days_ = tt.transport_traffic_days_;
  rtt.bitfields_ = tt.bitfields_;
  rtt.base_day_ = date::sys_days{2019_y / May / 3};
  rtt.base_day_idx_ = tt.day_idx(rtt.base_day_);
  rtt.location_rt_transports_.resize(tt.n_locations());

  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 4} + 9h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_start_time("00:30:00");
  td->set_start_date("20190503");
  td->set_trip_id("T_RE2");

  e->mutable_trip_update()->mutable_trip()->CopyFrom(*td);
  {
    auto const stop_update = e->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->mutable_departure()->set_delay(2 * 60 * 60 /* 2h */);
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);

  auto const results =
      raptor_search(tt, &rtt, "A", "D", sys_days{May / 2 / 2019} + 23h);
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  EXPECT_EQ(std::string_view{fwd_journeys}, ss.str());
}

constexpr auto const bwd_journeys = R"(
)";

TEST(routing, rt_raptor_backward) {
  timetable tt;
  rt_timetable rtt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const results = raptor_search(
      tt, &rtt, "0000003", "0000001",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
      direction::kBackward);

  ASSERT_EQ(2U, results.size());

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  EXPECT_EQ(std::string_view{bwd_journeys}, ss.str());
}
