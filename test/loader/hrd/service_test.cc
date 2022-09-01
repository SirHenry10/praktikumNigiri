#include "doctest/doctest.h"

#include <iostream>
#include <set>

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/print_transport.h"

#include "./hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

auto const expected = std::set<std::string>{R"(TRAFFIC_DAYS=0000100
2020-03-31 (day_idx=4)
 0: 0000003 C...............................................                               d: 31.03 02:05 [31.03 04:05]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/125/0000006/1510/, src=0}]
 1: 0000004 D............................................... a: 31.03 19:06 [31.03 21:06]  d: 31.03 20:07 [31.03 22:07]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/125/0000006/1510/, src=0}]
 2: 0000005 E............................................... a: 01.04 00:08 [01.04 02:08]  d: 01.04 00:09 [01.04 02:09]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/125/0000006/1510/, src=0}]
 3: 0000006 F............................................... a: 01.04 01:10 [01.04 03:10]
)",
                                            R"(TRAFFIC_DAYS=0000100
2020-03-31 (day_idx=4)
 0: 0000003 C...............................................                               d: 31.03 04:05 [31.03 06:05]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/245/0000006/1630/, src=0}]
 1: 0000004 D............................................... a: 31.03 21:06 [31.03 23:06]  d: 31.03 22:07 [01.04 00:07]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/245/0000006/1630/, src=0}]
 2: 0000005 E............................................... a: 01.04 02:08 [01.04 04:08]  d: 01.04 02:09 [01.04 04:09]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/245/0000006/1630/, src=0}]
 3: 0000006 F............................................... a: 01.04 03:10 [01.04 05:10]
)",
                                            R"(TRAFFIC_DAYS=0000100
2020-03-31 (day_idx=4)
 0: 0000003 C...............................................                               d: 31.03 06:05 [31.03 08:05]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/365/0000006/1750/, src=0}]
 1: 0000004 D............................................... a: 31.03 23:06 [01.04 01:06]  d: 01.04 00:07 [01.04 02:07]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/365/0000006/1750/, src=0}]
 2: 0000005 E............................................... a: 01.04 04:08 [01.04 06:08]  d: 01.04 04:09 [01.04 06:09]  [{name=ICE 1337, day=2020-03-31, id=1337/0000003/365/0000006/1750/, src=0}]
 3: 0000006 F............................................... a: 01.04 05:10 [01.04 07:10]
)",
                                            R"(TRAFFIC_DAYS=0001000
2020-03-30 (day_idx=3)
 0: 0000002 B...............................................                               d: 30.03 01:02 [30.03 03:02]  [{name=ICE 1337, day=2020-03-30, id=1337/0000002/62/0000005/1688/, src=0}]
 1: 0000003 C............................................... a: 30.03 05:04 [30.03 07:04]  d: 30.03 06:05 [30.03 08:05]  [{name=ICE 1337, day=2020-03-30, id=1337/0000002/62/0000005/1688/, src=0}]
 2: 0000004 D............................................... a: 30.03 23:06 [31.03 01:06]  d: 31.03 00:07 [31.03 02:07]  [{name=ICE 1337, day=2020-03-30, id=1337/0000002/62/0000005/1688/, src=0}]
 3: 0000005 E............................................... a: 31.03 04:08 [31.03 06:08]
)",
                                            R"(TRAFFIC_DAYS=0001000
2020-03-30 (day_idx=3)
 0: 0000007 G...............................................                               d: 30.03 00:30 [30.03 02:30]  [{name=RE 815, day=2020-03-30, id=815/0000007/30/0000006/1510/, src=0}]
 1: 0000003 C............................................... a: 30.03 01:20 [30.03 03:20]  d: 30.03 02:05 [30.03 04:05]  [{name=RE 815, day=2020-03-30, id=815/0000007/30/0000006/1510/, src=0}]
 2: 0000004 D............................................... a: 30.03 19:06 [30.03 21:06]  d: 30.03 20:07 [30.03 22:07]  [{name=RE 815, day=2020-03-30, id=815/0000007/30/0000006/1510/, src=0}]
 3: 0000005 E............................................... a: 31.03 00:08 [31.03 02:08]  d: 31.03 00:09 [31.03 02:09]  [{name=RE 815, day=2020-03-30, id=815/0000007/30/0000006/1510/, src=0}]
 4: 0000006 F............................................... a: 31.03 01:10 [31.03 03:10]
)",
                                            R"(TRAFFIC_DAYS=0010000
2020-03-29 (day_idx=2)
 0: 0000002 B...............................................                               d: 29.03 21:02 [29.03 23:02]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1262/0000005/2888/, src=0}]
 1: 0000003 C............................................... a: 30.03 01:04 [30.03 03:04]  d: 30.03 02:05 [30.03 04:05]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1262/0000005/2888/, src=0}]
 2: 0000004 D............................................... a: 30.03 19:06 [30.03 21:06]  d: 30.03 20:07 [30.03 22:07]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1262/0000005/2888/, src=0}]
 3: 0000005 E............................................... a: 31.03 00:08 [31.03 02:08]
)",
                                            R"(TRAFFIC_DAYS=0010000
2020-03-29 (day_idx=2)
 0: 0000002 B...............................................                               d: 29.03 23:02 [30.03 01:02]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1382/0000005/3008/, src=0}]
 1: 0000003 C............................................... a: 30.03 03:04 [30.03 05:04]  d: 30.03 04:05 [30.03 06:05]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1382/0000005/3008/, src=0}]
 2: 0000004 D............................................... a: 30.03 21:06 [30.03 23:06]  d: 30.03 22:07 [31.03 00:07]  [{name=ICE 1337, day=2020-03-29, id=1337/0000002/1382/0000005/3008/, src=0}]
 3: 0000005 E............................................... a: 31.03 02:08 [31.03 04:08]
)",
                                            R"(TRAFFIC_DAYS=0010000
2020-03-29 (day_idx=2)
 0: 0000008 H...............................................                               d: 29.03 22:30 [30.03 00:30]  [{name=IC 3374, day=2020-03-29, id=3374/0000008/1350/0000006/2950/, src=0}]
 1: 0000007 G............................................... a: 29.03 22:43 [30.03 00:43]  d: 29.03 22:45 [30.03 00:45]  [{name=IC 3374, day=2020-03-29, id=3374/0000008/1350/0000006/2950/, src=0}]
 2: 0000009 I............................................... a: 30.03 23:24 [31.03 01:24]  d: 30.03 23:25 [31.03 01:25]  [{name=IC 3374, day=2020-03-29, id=3374/0000008/1350/0000006/2950/, src=0}]
 3: 0000005 E............................................... a: 30.03 23:55 [31.03 01:55]  d: 31.03 01:09 [31.03 03:09]  [{name=IC 3374, day=2020-03-29, id=3374/0000008/1350/0000006/2950/, src=0}]
 4: 0000006 F............................................... a: 31.03 01:10 [31.03 03:10]
)",
                                            R"(TRAFFIC_DAYS=0100000
2020-03-28 (day_idx=1)
 0: 0000001 A...............................................                               d: 28.03 21:00 [28.03 22:00]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1260/0000004/2586/, src=0}]
 1: 0000002 B............................................... a: 28.03 22:01 [28.03 23:01]  d: 28.03 22:02 [28.03 23:02]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1260/0000004/2586/, src=0}]
 2: 0000003 C............................................... a: 29.03 01:04 [29.03 03:04]  d: 29.03 02:05 [29.03 04:05]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1260/0000004/2586/, src=0}]
 3: 0000004 D............................................... a: 29.03 19:06 [29.03 21:06]
)",
                                            R"(TRAFFIC_DAYS=0100000
2020-03-28 (day_idx=1)
 0: 0000001 A...............................................                               d: 28.03 23:00 [29.03 00:00]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1380/0000004/2706/, src=0}]
 1: 0000002 B............................................... a: 29.03 00:01 [29.03 01:01]  d: 29.03 00:02 [29.03 01:02]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1380/0000004/2706/, src=0}]
 2: 0000003 C............................................... a: 29.03 03:04 [29.03 05:04]  d: 29.03 04:05 [29.03 06:05]  [{name=ICE 1337, day=2020-03-28, id=1337/0000001/1380/0000004/2706/, src=0}]
 3: 0000004 D............................................... a: 29.03 21:06 [29.03 23:06]
)",
                                            R"(TRAFFIC_DAYS=0100000
2020-03-28 (day_idx=1)
 0: 0000008 H...............................................                               d: 28.03 23:30 [29.03 00:30]  [{name=IC 3374, day=2020-03-28, id=3374/0000008/1410/0000006/2950/, src=0}]
 1: 0000007 G............................................... a: 28.03 23:43 [29.03 00:43]  d: 28.03 23:45 [29.03 00:45]  [{name=IC 3374, day=2020-03-28, id=3374/0000008/1410/0000006/2950/, src=0}]
 2: 0000009 I............................................... a: 29.03 23:24 [30.03 01:24]  d: 29.03 23:25 [30.03 01:25]  [{name=IC 3374, day=2020-03-28, id=3374/0000008/1410/0000006/2950/, src=0}]
 3: 0000005 E............................................... a: 29.03 23:55 [30.03 01:55]  d: 30.03 01:09 [30.03 03:09]  [{name=IC 3374, day=2020-03-28, id=3374/0000008/1410/0000006/2950/, src=0}]
 4: 0000006 F............................................... a: 30.03 01:10 [30.03 03:10]
)",
                                            R"(TRAFFIC_DAYS=1000000
2020-03-27 (day_idx=0)
 0: 0000008 H...............................................                               d: 27.03 23:30 [28.03 00:30]  [{name=IC 3374, day=2020-03-27, id=3374/0000008/1410/0000006/2950/, src=0}]
 1: 0000007 G............................................... a: 27.03 23:43 [28.03 00:43]  d: 27.03 23:45 [28.03 00:45]  [{name=IC 3374, day=2020-03-27, id=3374/0000008/1410/0000006/2950/, src=0}]
 2: 0000009 I............................................... a: 29.03 00:24 [29.03 01:24]  d: 29.03 00:25 [29.03 01:25]  [{name=IC 3374, day=2020-03-27, id=3374/0000008/1410/0000006/2950/, src=0}]
 3: 0000005 E............................................... a: 29.03 00:55 [29.03 01:55]  d: 29.03 01:09 [29.03 03:09]  [{name=IC 3374, day=2020-03-27, id=3374/0000008/1410/0000006/2950/, src=0}]
 4: 0000006 F............................................... a: 29.03 01:10 [29.03 03:10]
)"};

std::set<std::string> service_strings(timetable const& tt) {
  auto const reverse = [](std::string s) {
    std::reverse(s.begin(), s.end());
    return s;
  };
  auto const num_days = static_cast<size_t>(
      (tt.date_range_.to_ - tt.date_range_.from_ + 1_days) / 1_days);
  auto ret = std::set<std::string>{};
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    std::stringstream out;
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRAFFIC_DAYS="
        << reverse(
               traffic_days.to_string().substr(traffic_days.size() - num_days))
        << "\n";
    for (auto d = tt.date_range_.from_; d != tt.date_range_.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - tt.date_range_.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, {transport_idx, day_idx});
      }
    }
    ret.emplace(out.str());
  }
  return ret;
}

TEST_CASE("a") {
  auto tt = std::make_shared<timetable>();
  load_timetable(source_idx_t{0U}, hrd_5_20_26,
                 nigiri::test_data::hrd_timetable::files(), *tt);

  std::cout << "OUTPUT:\n";
  std::cout << "constexpr auto const expected = std::set<std::string>{";
  for (auto const& ss : service_strings(*tt)) {
    std::cout << "R\"(" << ss << ")\",\n";
  }
  std::cout << "};\n";

  std::cout << "\n\n\nEXPECTED:\n";
  std::cout << "constexpr auto const expected = std::set<std::string>{";
  for (auto const& ss : expected) {
    std::cout << "R\"(" << ss << ")\"\n";
  }
  std::cout << "};\n";

  CHECK(expected == service_strings(*tt));
}
