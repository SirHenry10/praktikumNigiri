#pragma once

#include <vector>

#include "fmt/core.h"

#include "date/date.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

namespace nigiri::rt::test {

struct trip {
  struct delay {
    std::string stop_id_;
    event_type ev_type_;
    unsigned delay_minutes_{0U};
  };
  std::string trip_id_;
  std::vector<delay> delays_;
};

template <typename T>
std::int64_t to_unix(T&& x) {
  return std::chrono::time_point_cast<std::chrono::seconds>(x)
      .time_since_epoch()
      .count();
};

transit_realtime::FeedMessage to_feed_msg(std::vector<trip> const& trip_delays,
                                          date::sys_seconds const msg_time);

}  // namespace nigiri::rt::test