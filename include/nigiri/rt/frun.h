#pragma once

#include <iosfwd>

#include "nigiri/location.h"
#include "nigiri/rt/run.h"
#include "nigiri/stop.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

// Full run. Same as `run` data structure, extended with timetable and
// rt_timetable to be able to look up additional info.
struct frun : public run {
  struct run_stop {
    stop get_stop() const noexcept;
    location get_location() const noexcept;
    location_idx_t get_location_idx() const noexcept;
    std::string_view get_location_name() const noexcept;
    std::string_view get_location_track() const noexcept;
    unixtime_t scheduled_time(event_type const ev_type) const noexcept;
    unixtime_t time(event_type const ev_type) const noexcept;
    std::string_view line() const noexcept;
    std::string_view scheduled_line() const noexcept;
    bool in_allowed() const noexcept;
    bool out_allowed() const noexcept;
    bool operator==(run_stop const&) const = default;
    friend std::ostream& operator<<(std::ostream&, run_stop const&);

    frun const* fr_{nullptr};
    stop_idx_t stop_idx_{0U};
  };

  struct iterator {
    using difference_type = stop_idx_t;
    using value_type = run_stop;
    using pointer = run_stop;
    using reference = run_stop;
    using iterator_category = std::forward_iterator_tag;

    iterator& operator++() noexcept;
    iterator operator++(int) noexcept;

    bool operator==(iterator const o) const noexcept;
    bool operator!=(iterator o) const noexcept;

    run_stop operator*() const noexcept;

    run_stop rs_;
  };
  using const_iterator = iterator;

  std::string_view name() const noexcept;
  debug dbg() const noexcept;

  frun(timetable const&, rt_timetable const*, run);
  frun(timetable const&, rt_timetable const&, rt_transport_idx_t const);
  frun(timetable const&, rt_timetable const*, transport const);

  iterator begin() const noexcept;
  iterator end() const noexcept;

  friend iterator begin(frun const& fr) noexcept;
  friend iterator end(frun const& fr) noexcept;

  stop_idx_t size() const noexcept;

  run_stop operator[](stop_idx_t) const noexcept;

  void print(std::ostream&, interval<stop_idx_t> stop_range);
  friend std::ostream& operator<<(std::ostream&, frun const&);

  timetable const* tt_;
  rt_timetable const* rtt_;
};

}  // namespace nigiri::rt