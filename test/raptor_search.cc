#include "./raptor_search.h"

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"
#include "nigiri/routing/gpu_raptor_translator.h"

namespace nigiri::test {

unixtime_t parse_time(std::string_view s, char const* format) {
  std::stringstream in;
  in << s;

  date::local_seconds ls;
  std::string tz;
  in >> date::parse(format, ls, tz);

  return std::chrono::time_point_cast<unixtime_t::duration>(
      date::make_zoned(tz, ls).get_sys_time());
}

template <direction SearchDir>
pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           GPU gpu = nonGPU) {
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  if (rtt == nullptr) {
    if(gpu == useGPU){
      std::cerr << "debugger: 0" << std::endl;
      //gpu_raptor without rtt
      using algo_t = gpu_raptor_translator<SearchDir, false>;
      return *(routing::search<SearchDir, algo_t>{tt, rtt, search_state,
                                                  algo_state, std::move(q)}
                   .execute()
                   .journeys_);
    }else{
    using algo_t = routing::raptor<SearchDir, false>;
    return *(routing::search<SearchDir, algo_t>{tt, rtt, search_state,
                                                algo_state, std::move(q)}
                 .execute()
                 .journeys_);
    }
  }else{
    if(gpu == useGPU){
      //current state not working
      using algo_t = gpu_raptor_translator<SearchDir, true>;
      return *(routing::search<SearchDir, algo_t>{tt, rtt, search_state,
                                                  algo_state, std::move(q)}
                   .execute()
                   .journeys_);
    }else {
      using algo_t = routing::raptor<SearchDir, true>;
      return *(routing::search<SearchDir, algo_t>{tt, rtt, search_state,
                                                  algo_state, std::move(q)}
                   .execute()
                   .journeys_);
    }
  }
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir) {
  if (search_dir == direction::kForward) {
    return raptor_search<direction::kForward>(tt, rtt, std::move(q));
  } else {
    return raptor_search<direction::kBackward>(tt, rtt, std::move(q));
  }
}
pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t time,
                                           direction const search_dir,
                                           routing::clasz_mask_t const mask) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .prf_idx_ = 0,
      .allowed_claszes_ = mask};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           direction const search_dir,
                                           routing::clasz_mask_t mask) {
  return raptor_search(tt, rtt, from, to, parse_time(time, "%Y-%m-%d %H:%M %Z"),
                       search_dir, mask);
}

pareto_set<routing::journey> raptor_intermodal_search(
    timetable const& tt,
    rt_timetable const* rtt,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    routing::start_time_t const interval,
    direction const search_dir,
    std::uint8_t const min_connection_count,
    bool const extend_interval_earlier,
    bool const extend_interval_later) {
  auto q = routing::query{
      .start_time_ = interval,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later,
      .prf_idx_ = 0};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           GPU gpu,
                                           direction const search_dir) {
  std::cerr << "debugger: -1" << std::endl;
  if (search_dir == direction::kForward) {
    return raptor_search<direction::kForward>(tt, rtt, std::move(q),gpu);
  } else {
    return raptor_search<direction::kBackward>(tt, rtt, std::move(q),gpu);
  }
}
//Hier rein
pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t time,
                                           GPU gpu,
                                           direction const search_dir,
                                           routing::clasz_mask_t const mask) {
  std::cerr << "raptor_search" << std::endl;
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .prf_idx_ = 0,
      .allowed_claszes_ = mask};
  return raptor_search(tt, rtt, std::move(q),gpu, search_dir);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           GPU gpu,
                                           direction const search_dir,
                                           routing::clasz_mask_t mask) {
  return raptor_search(tt, rtt, from, to, parse_time(time, "%Y-%m-%d %H:%M %Z"),
                       gpu,search_dir, mask);
}

pareto_set<routing::journey> raptor_intermodal_search(
    timetable const& tt,
    rt_timetable const* rtt,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    routing::start_time_t const interval,
    GPU gpu,
    direction const search_dir,
    std::uint8_t const min_connection_count,
    bool const extend_interval_earlier,
    bool const extend_interval_later) {
  auto q = routing::query{
      .start_time_ = interval,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later,
      .prf_idx_ = 0};
  return raptor_search(tt, rtt, std::move(q),gpu, search_dir);
}
}  // namespace nigiri::test