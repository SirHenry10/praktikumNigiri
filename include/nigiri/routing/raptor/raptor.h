#pragma once

#include "utl/enumerate.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/delta_t.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "debug.h"

namespace nigiri::routing {

struct raptor_stats {
  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};

template <direction SearchDir>
struct raptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  raptor(timetable const& tt,
         raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         day_idx_t const base)
      : tt_{tt},
        state_{state},
        is_dest_{is_dest},
        dist_to_end_{dist_to_dest},
        lb_{lb},
        base_{base},
        n_days_{tt_.internal_interval_days().size().count()} {
    reset_state();
  }

  algo_stats_t get_stats() const { return stats_; }

  void reset_state() {
    state_.reset(tt_.n_locations(), tt_.n_routes(), kInvalid);
    utl::fill(time_at_dest_, kInvalid);
  }

  void reset_arrivals() {
    utl::fill(state_.best_, kInvalid);
    utl::fill(state_.tmp_, kInvalid);
    utl::fill(time_at_dest_, kInvalid);
    state_.round_times_.reset(kInvalid);
    time_at_dest_ = kInvalid;
  }

  void init(location_idx_t const l, unixtime_t const t) {
    state_.round_times_[0U][to_idx(l)] = unix_to_delta(base(), t);
    state_.station_mark_[to_idx(l)] = true;
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               pareto_set<journey>& results) {
    utl::fill(state_.best_, kInvalid);  // TODO(felix) eval
    utl::fill(state_.tmp_, kInvalid);

    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;

    trace("worst_time_at_dest={} -> time_at_dest=best({}, {})={}={}\n",
          worst_time_at_dest, to_unix(time_at_dest_),
          get_best(unix_to_delta(base(), worst_time_at_dest), time_at_dest_),
          delta_to_unix(base(),
                        get_best(unix_to_delta(base(), worst_time_at_dest),
                                 time_at_dest_)),
          get_best(unix_to_delta(base(), worst_time_at_dest), time_at_dest_));

    time_at_dest_ =
        get_best(unix_to_delta(base(), worst_time_at_dest), time_at_dest_);

    trace_print_init_state();

    for (auto k = 1U; k != end_k; ++k) {
      set_time_at_destination(k);

      auto any_marked = false;
      for (auto i = 0U; i != tt_.n_locations(); ++i) {
        if (state_.station_mark_[i]) {
          any_marked = true;
          for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
            state_.route_mark_[to_idx(r)] = true;
          }
        }
      }

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_, false);

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      any_marked = false;
      for (auto r_id = 0U; r_id != tt_.n_routes(); ++r_id) {
        if (state_.route_mark_[r_id]) {
          ++stats_.n_routes_visited_;
          trace("┊ ├k={} updating route {}\n", k, r_id);
          any_marked |= update_route(k, route_idx_t{r_id});
        }
      }

      utl::fill(state_.route_mark_, false);

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_, false);

      update_transfers(k);
      update_footpaths(k);
      update_intermodal_footpaths(k);

      trace_print_state_after_round();
    }

    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      auto const is_dest = is_dest_[i];
      if (!is_dest) {
        continue;
      }

      for (auto k = 1U; k != end_k; ++k) {
        auto const dest_time = state_.round_times_[k][i];
        if (dest_time != kInvalid) {
          trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                start_time, delta_to_unix(base(), state_.round_times_[k][i]),
                location{tt_, location_idx_t{i}}, k - 1);
          auto const [optimal, it, dominated_by] = results.add(
              journey{.legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(base(), dest_time),
                      .dest_ = location_idx_t{i},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
          if (!optimal) {
            trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                  dominated_by->start_time_, dominated_by->dest_time_,
                  location{tt_, dominated_by->dest_}, dominated_by->transfers_);
          }
        }
      }
    }
  }

  void reconstruct(query const& q, journey& j) {
    reconstruct_journey<SearchDir>(tt_, q, state_, j, base(), base_);
  }

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + to_idx(base_) * date::days{1};
  }

  void set_time_at_destination(unsigned const k) {
    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      if (is_dest_[i]) {
        time_at_dest_ =
            get_best(time_at_dest_, state_.round_times_[k][i], state_.best_[i]);
      }
    }
  }

  void update_transfers(unsigned const k) {
    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      if (!state_.prev_station_mark_[i]) {
        continue;
      }
      auto const is_dest = is_dest_[i];
      auto const transfer_time =
          (!is_intermodal_dest() && is_dest)
              ? 0
              : dir(tt_.locations_.transfer_time_[location_idx_t{i}]).count();
      auto const fp_target_time =
          static_cast<delta_t>(state_.tmp_[i] + transfer_time);
      if (is_better(fp_target_time,
                    state_.best_[i]) && /* TODO _or_eq?? pretrip */
          is_better_or_eq(fp_target_time, time_at_dest_)) {
        if (lb_[i] == kUnreachable ||
            !is_better_or_eq(fp_target_time + dir(lb_[i]),
                             time_at_dest_) /*  TODO _or_eq?? pretrip*/) {
          ++stats_.fp_update_prevented_by_lower_bound_;
          continue;
        }

        ++stats_.n_earliest_arrival_updated_by_footpath_;
        state_.round_times_[k][i] = fp_target_time;
        state_.best_[i] = fp_target_time;
        state_.station_mark_[i] = true;
        if (is_dest) {
          time_at_dest_ = fp_target_time;
        }
      }
    }
  }

  void update_footpaths(unsigned const k) {
    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      if (!state_.prev_station_mark_[i]) {
        continue;
      }

      auto const l_idx = location_idx_t{i};
      auto const& fps = kFwd ? tt_.locations_.footpaths_out_[l_idx]
                             : tt_.locations_.footpaths_in_[l_idx];
      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;

        auto const target = to_idx(fp.target_);
        auto const fp_target_time =
            clamp(state_.tmp_[i] + dir(fp.duration_).count());

        if (is_better(fp_target_time, state_.best_[target]) &&
            is_better_or_eq(fp_target_time,
                            time_at_dest_) /* TODO _or_eq pretrip?? */) {
          auto const lower_bound = lb_[to_idx(fp.target_)];
          if (lower_bound == kUnreachable ||
              !is_better_or_eq(fp_target_time + dir(lower_bound),
                               time_at_dest_) /* TODO _or_eq pretrip?? */) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            trace_upd(
                "┊ ├k={} *** LB NO UPD: (from={}, tmp={}) --{}--> (to={}, "
                "best={}) --> update => {}, LB={}, LB_AT_DEST={}, DEST={}\n",
                k, location{tt_, l_idx}, to_unix(state_.tmp_[to_idx(l_idx)]),
                fp.duration_, location{tt_, fp.target_},
                state_.best_[to_idx(fp.target_)], fp_target_time, lower_bound,
                to_unix(clamp(fp_target_time + dir(lower_bound))),
                to_unix(time_at_dest_));
            continue;
          }

          trace_upd(
              "┊ ├k={}   footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
              "update => {}\n",
              k, location{tt_, l_idx}, to_unix(state_.tmp_[to_idx(l_idx)]),
              fp.duration_, location{tt_, fp.target_},
              to_unix(state_.best_[to_idx(fp.target_)]), fp_target_time);

          ++stats_.n_earliest_arrival_updated_by_footpath_;
          state_.round_times_[k][to_idx(fp.target_)] = fp_target_time;
          state_.best_[to_idx(fp.target_)] = fp_target_time;
          state_.station_mark_[to_idx(fp.target_)] = true;
          if (is_dest_[to_idx(fp.target_)]) {
            time_at_dest_ = fp_target_time;
          }
        } else {
          trace(
              "┊ ├k={}   NO FP UPDATE: {} [best={}] --{}--> {} "
              "[best={}, time_at_dest={}]\n",
              k, location{tt_, l_idx}, state_.best_[to_idx(l_idx)],
              fp.duration_, location{tt_, fp.target_},
              state_.best_[to_idx(fp.target_)], to_unix(time_at_dest_));
        }
      }
    }
  }

  void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      if ((state_.prev_station_mark_[i] || state_.station_mark_[i]) &&
          dist_to_end_[i] != kUnreachable) {
        auto const end_time = clamp(get_best(state_.best_[i], state_.tmp_[i]) +
                                    dir(dist_to_end_[i]));

        if (is_better(end_time, state_.best_[kIntermodalTarget])) {
          state_.round_times_[k][kIntermodalTarget] = end_time;
          state_.best_[kIntermodalTarget] = end_time;
          time_at_dest_ = end_time;
        }

        trace("┊ │k={}  INTERMODAL FOOTPATH: location={}, dist_to_end={}\n", k,
              location{tt_, location_idx_t{i}}, dist_to_end_[i]);
      }
    }
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto et = transport{};
    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<unsigned>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stop = timetable::stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stop.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      if (!et.is_valid() && !state_.prev_station_mark_[l_idx]) {
        trace("┊ │k={}  stop_idx={} {}: not marked, no et - skip\n", k,
              stop_idx, location{tt_, location_idx_t{l_idx}});
        continue;
      }

      trace(
          "┊ │k={}  stop_idx={}, location={}, round_times={}, best={}, "
          "tmp={}\n",
          k, stop_idx, location{tt_, stop.location_idx()},
          to_unix(state_.round_times_[k - 1][l_idx]),
          to_unix(state_.best_[l_idx]), to_unix(state_.tmp_[l_idx]));

      auto current_best = kInvalid;
      if (et.is_valid() && (kFwd ? stop.out_allowed() : stop.in_allowed())) {
        auto const by_transport = time_at_stop(
            r, et, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        current_best = get_best(state_.round_times_[k - 1][l_idx],
                                state_.tmp_[l_idx], state_.best_[l_idx]);
        /* TODO(felix) check
        if (state_.best_k_[l_idx] <= k) {
          current_best = get_best(current_best, );
        }
        */
        if (is_better(by_transport, current_best) &&
            is_better_or_eq(by_transport,
                            time_at_dest_) &&  // TODO _or_eq?? pretrip
            lb_[l_idx] != kUnreachable &&
            is_better_or_eq(by_transport + dir(lb_[l_idx]),
                            time_at_dest_) /* TODO _or_eq?? pretrip */) {
          trace_upd(
              "┊ │k={}    name={}, dbg={}, time_by_transport={}, BETTER THAN "
              "current_best={} => update, {} marking station {}!\n",
              k, tt_.transport_name(et.t_idx_), tt_.dbg(et.t_idx_),
              by_transport, current_best,
              !is_better(by_transport, current_best) ? "NOT" : "",
              location{tt_, stop.location_idx()});

          ++stats_.n_earliest_arrival_updated_by_route_;
          state_.tmp_[l_idx] = get_best(by_transport, state_.tmp_[l_idx]);
          state_.station_mark_[l_idx] = true;
          current_best = by_transport;
          any_marked = true;
        } else {
          trace_upd(
              "┊ │k={}    *** NO UPD: at={}, name={}, dbg={}, "
              "time_by_transport={}, current_best=min({}, {}, {})={} => {} - "
              "LB={}, LB_AT_DEST={}, TIME_AT_DEST={} "
              "(is_better(by_transport={}={}, current_best={}={})={}, "
              "is_better(by_transport={}={}, time_at_dest_={}={})={}, "
              "reachable={}, "
              "is_better(lb={}={}, time_at_dest_={}={})={})!\n",
              k, location{tt_, location_idx_t{l_idx}},
              tt_.transport_name(et.t_idx_), tt_.dbg(et.t_idx_),
              to_unix(by_transport), to_unix(state_.round_times_[k - 1][l_idx]),
              to_unix(state_.best_[l_idx]), to_unix(state_.tmp_[l_idx]),
              to_unix(current_best), location{tt_, location_idx_t{l_idx}},
              lb_[l_idx], to_unix(time_at_dest_),
              to_unix(clamp(by_transport + dir(lb_[l_idx]))), by_transport,
              to_unix(by_transport), current_best, to_unix(current_best),
              is_better(by_transport, current_best), by_transport,
              to_unix(by_transport), time_at_dest_, to_unix(time_at_dest_),
              is_better(by_transport, time_at_dest_),
              lb_[l_idx] != kUnreachable, by_transport + dir(lb_[l_idx]),
              to_unix(clamp(by_transport + dir(lb_[l_idx]))), time_at_dest_,
              to_unix(time_at_dest_), to_unix(time_at_dest_),
              is_better(clamp(by_transport + dir(lb_[l_idx])), time_at_dest_));
        }
      }

      if (is_last || !(kFwd ? stop.in_allowed() : stop.out_allowed()) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      auto const et_time_at_stop =
          et.is_valid()
              ? time_at_stop(r, et, stop_idx,
                             kFwd ? event_type::kDep : event_type::kArr)
              : kInvalid;
      auto const prev_round_time = state_.round_times_[k - 1][l_idx];
      if (is_better_or_eq(prev_round_time, et_time_at_stop)) {
        auto const [day, mam] = split(prev_round_time);
        auto const new_et = get_earliest_transport(r, stop_idx, day, mam);
        current_best =
            get_best(current_best, state_.best_[l_idx], state_.tmp_[l_idx]);
        if (new_et.is_valid() &&
            (current_best == kInvalid ||
             is_better_or_eq(
                 time_at_stop(r, new_et, stop_idx,
                              kFwd ? event_type::kDep : event_type::kArr),
                 et_time_at_stop))) {
          et = new_et;
        } else if (new_et.is_valid()) {
          trace("┊ │k={}    update et: no update time_at_stop={}\n", k,
                et_time_at_stop);
        }
      }
    }
    return any_marked;
  }

  transport get_earliest_transport(route_idx_t const r,
                                   unsigned const stop_idx,
                                   day_idx_t const day_at_stop,
                                   minutes_after_midnight_t const mam_at_stop) {
    ++stats_.n_earliest_trip_calls_;

    auto const n_days_to_iterate =
        std::min(kMaxTravelTime.count() / 1440U + 1,
                 kFwd ? static_cast<unsigned>(n_days_) - to_idx(day_at_stop)
                      : to_idx(day_at_stop) + 1U);

    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&, mam_at_stop = mam_at_stop]() {
      return std::lower_bound(
          get_begin_it(event_times), get_end_it(event_times), mam_at_stop,
          [&](auto&& a, auto&& b) { return is_better(a, b); });
    };

#if defined(NIGIRI_TRACING)
    auto const l_idx =
        timetable::stop{tt_.route_location_seq_[r][stop_idx]}.location_idx();

    trace(
        "┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}, "
        "n_days_to_iterate={}\n",
        '_', tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
        location{tt_, l_idx}, n_days_to_iterate);
#endif

    for (auto i = day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      auto const ev_time_range = it_range{
          i == 0U && kFwd ? seek_first_day() : get_begin_it(event_times),
          get_end_it(event_times)};
      if (ev_time_range.empty()) {
        continue;
      }

      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = minutes_after_midnight_t{ev.count() % 1440};

        if (is_better_or_eq(time_at_dest_, to_delta(day, ev_mam))) {
          trace(
              "┊ │k={}      => name={}, dbg={}, day={}={}, best_mam={}, "
              "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
              "BETTER!\n",
              '_', tt_.transport_name(tt_.route_transport_ranges_[r][t_offset]),
              tt_.dbg(tt_.route_transport_ranges_[r][t_offset]), day,
              tt_.to_unixtime(day, 0_minutes), mam_at_stop, ev_mam,
              tt_.to_unixtime(day, ev_mam), to_unix(time_at_dest_));
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (day == day_at_stop && !is_better_or_eq(mam_at_stop, ev_mam)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO REACH!\n",
              '_', t, tt_.transport_name(t), tt_.dbg(t), i, day, mam_at_stop,
              ev_mam, ev);
          continue;
        }

        auto const ev_day_offset =
            static_cast<day_idx_t::value_t>(ev.count() / 1440);
        auto const start_day =
            static_cast<std::size_t>(to_idx(day) - ev_day_offset);
        if (!tt_.bitfields_[tt_.transport_traffic_days_[t]].test(start_day)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "ev_day_offset={}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
              '_', t, tt_.transport_name(t), tt_.dbg(t), i, day, ev_day_offset,
              mam_at_stop, ev_mam, ev);
          continue;
        }

        trace(
            "┊ │k={}      => ET FOUND: name={}, dbg={}, at day {} "
            "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
            '_', tt_.transport_name(t), tt_.dbg(t), day, ev_day_offset, ev_mam,
            ev, tt_.to_unixtime(day, ev_mam));
        return {t, static_cast<day_idx_t>(day - ev_day_offset)};
      }
    }
    return {};
  }

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       unsigned const stop_idx,
                       event_type const ev_type) {
    trace(
        "time at stop: {}\n",
        tt_.to_unixtime(t.day_, tt_.event_mam(r, t.t_idx_, stop_idx, ev_type)));
    return to_delta(t.day_, tt_.event_mam(r, t.t_idx_, stop_idx, ev_type));
  }

  delta_t to_delta(day_idx_t const day, minutes_after_midnight_t const mam) {
    auto const day_idx = static_cast<int>(to_idx(day));
    auto const base_idx = static_cast<int>(to_idx(base_));

    trace("to delta: day={}, base={}={}, mam={}, {}={} = {}\n", day_idx,
          base_idx, base(), mam.count(),
          (day_idx - base_idx) * 1440 + mam.count(), tt_.to_unixtime(day, mam),
          clamp((day_idx - base_idx) * 1440 + mam.count()));

    return clamp((day_idx - base_idx) * 1440 + mam.count());
  }

  unixtime_t to_unix(delta_t const t) { return delta_to_unix(base(), t); }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
    return split_day_mam(base_, x);
  }

  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }

  void update_time_at_dest(unsigned const k, delta_t const t) {
    for (auto i = k; i != time_at_dest_.size(); ++i) {
      time_at_dest_[i] = get_best(time_at_dest_[i], t);
    }
  }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  timetable const& tt_;
  raptor_state& state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  int n_days_;
  raptor_stats stats_;
};

}  // namespace nigiri::routing