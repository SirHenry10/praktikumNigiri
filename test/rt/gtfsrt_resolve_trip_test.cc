#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/timetable.h"

#include "nigiri/rt/gtfsrt_resolve_trip.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;

namespace {

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
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
S_RE1,20190503,1
S_RE2,20190504,1
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
T_RE2,00:30:00,00:30:00,C,2,0,0
)"}}}};
}

}  // namespace

TEST(rt, gtfsrt_resolve_trip) {
  timetable tt;
  rt_timetable rtt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);

  {  // test start time >24:00:00
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "49:00:00";
    *td.mutable_start_date() = "20190503";
    *td.mutable_trip_id() = "T_RE1";

    auto const t = rt::gtfsrt_resolve_trip(date::sys_days{2019_y / May / 3}, tt,
                                           rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.has_value());
  }

  {  // test start time that's on the prev. day in UTC
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "00:30:00";
    *td.mutable_start_date() = "20190504";
    *td.mutable_trip_id() = "T_RE2";

    auto const t = rt::gtfsrt_resolve_trip(date::sys_days{2019_y / May / 4}, tt,
                                           rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.has_value());
  }

  {  // test without start_time and start_date (assuming "today" as date)
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_trip_id() = "T_RE2";

    // 2019-05-03 00:30 CEST is
    // 2019-05-02 21:30 UTC
    // -> we give "today" in UTC (start_day would be local days)
    auto const t = rt::gtfsrt_resolve_trip(date::sys_days{2019_y / May / 2}, tt,
                                           rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.has_value());
  }
}