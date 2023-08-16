#pragma once

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include "utl/erase_if.h"

namespace nigiri::routing {
struct station_filter {

  struct weight_info {
    location_idx_t l_;
    int weight_;
  };

/*static vector_map<location_idx_t, vector<std::string>> create_names_lists(std::vector<start>& starts, timetable const& tt) {
  printf("in create names\n");
  vector_map<location_idx_t, vector<std::string>> names_lists;
  for(auto const& s : starts) {
    vector<std::string> names;
    location_idx_t l = s.stop_;
    auto const vr = tt.location_routes_.at(l);
    for(auto const ri : vr) {
      auto const& transport_range = tt.route_transport_ranges_[route_idx_t{ri}];
      std::string_view sv_tname = tt.transport_name(transport_range.from_);
      std::string tname = {sv_tname.begin(), sv_tname.end()};
      // tname = "Bus RE 2" od "ICE 1337" --> dann wäre aufteilen falsch
      while(tname.contains(" ")) {
        int indexleer = tname.find(" ");
        std::string safename = tname.substr(0, indexleer);
        tname = tname.substr(indexleer+1);
        names.emplace_back(safename);
      }
      names.emplace_back(tname); // checked
    }
    names_lists.emplace_back(names); // hier stimmt was nicht
    //names_lists.at(l) = names;
  }
  printf("names_list: %u\n", names_lists.size()); // 1
  return names_lists;
}*/

  static void percentage_filter(std::vector<start>& starts, double percent) {
    printf("in percentage_filter\n");
    auto const min = [&](start const& a, start const& b) {
      return a.time_at_stop_ < b.time_at_stop_;
    };
    std::sort(starts.begin(), starts.end(), min);
    size_t percent_to_dismiss = static_cast<size_t>(starts.size() * percent);
    size_t new_size = starts.size() - percent_to_dismiss;
    if(starts.at(starts.size()-1).time_at_stop_ == starts.at(0).time_at_stop_) {
      return;
    }
    starts.resize(new_size);
    starts.shrink_to_fit();
  }

  static void weighted_filter(std::vector<start>& starts, timetable const& tt, bool linefilter) {
    printf("in weighted_filter\n");
    vector<weight_info> v_weights;
    int most = 1;
    auto const weighted = [&](start const& a) {
      location_idx_t l = a.stop_;
      for(auto const w : v_weights) {
        if(l == w.l_) {
          double percent = w.weight_ * 100.0 / most;
          if(percent > 22.0) {
            return false;
          }
          else return true;
        }
      }
      return true;
    };
    // Example:
    // dep_count = 3, local_count=2, slow_count=1*2=2,    o = 5   -> weight = 8,3
    // dep_count = 2, slow_count=1*2=2, fast_count=1*3=3, o = 10  -> weight = 7,2
    // 6 = 0-3min; 5 = 3-5min; 4 = 5-7min; 3 = 7-10min; 2 = 10-15min; 1 = 15-20min; 0 = > 20min
    for (auto const& s : starts) {
      auto const l = s.stop_;
      auto const o = s.time_at_stop_ - s.time_at_start_;
      size_t dep_count = tt.depature_count_.at(l);
      int local_count = tt.get_groupclass_count(l, group::klocal);
      int slow_count = tt.get_groupclass_count(l, group::kslow) * 2;
      int fast_count = tt.get_groupclass_count(l, group::kfast) * 3;
      int weight = local_count + slow_count + fast_count + (dep_count/10);
      if(o.count() >= 15 && o.count() < 20) weight += 1;
      if(o.count() >= 10 && o.count() < 15) weight += 2;
      if(o.count() >= 7 && o.count() < 10) weight += 3;
      if(o.count() >= 5 && o.count() < 7) weight += 4;
      if(o.count() >= 3 && o.count() < 5) weight += 5;
      if(o.count() >= 0 && o.count() < 3) weight += 6;
      int extra_weight = 0;
      if(linefilter) {
        extra_weight = line_filter(starts, tt, s);
      }
      weight_info wi = {l, (weight+extra_weight)};
      v_weights.emplace_back(wi);
      most = weight > most ? weight : most;
    }
    if(most == 1) {
      return;
    }
    utl::erase_if(starts, weighted);
  }

  static vector<location_idx_t> find_name(const vector<char>& find, timetable const& tt) {//,
                                          //vector_map<location_idx_t, vector<std::string>> in) {
    vector<location_idx_t> found_at;
    for(auto lidx = location_idx_t{0}; lidx < tt.names_lists_.size(); lidx++) {
      for(auto const p : tt.names_lists_.at(lidx)) {
        if(find == p) {
          found_at.emplace_back(lidx);
        }
      }
    }
    return found_at;
  }

  static start find_start_from_locidx(std::vector<start>& starts, location_idx_t locidx) {
    for(start s : starts) {
      if(s.stop_ == locidx) return s;
    }
    return start();
  }

  // TODO: überprüfen, ob die Namen richtig verwendet werden.
  static int line_filter(std::vector<start>& starts, timetable const& tt, start this_start) {
    //vector_map<location_idx_t, vector<std::string>> starts_all_names = create_names_lists(starts, tt);
    printf("in line_filter\n");
    duration_t o = this_start.time_at_stop_ - this_start.time_at_start_;
    auto const l = this_start.stop_;
    int weight_count = 0;
    duration_t dur_off;
    //vector<std::string> this_start_names = starts_all_names.at(l); // hier gehts schief
    vector<vector<char>> this_start_names = tt.names_lists_.at(l);
    vector<location_idx_t> v_li;
    for(auto const name : this_start_names) {
      printf("\t\t\t\t\t name: %s \n", name.data());
      v_li = find_name(name, tt); //starts_all_names
    }
    for(auto const a : v_li) {
      start s = find_start_from_locidx(starts, a);
      dur_off = s.time_at_stop_ - s.time_at_start_;
      if(o < dur_off) {
        weight_count++;
      }
    }
    // einzeln ergibt lineFilter keinen sinn, weil die Station noch andere Routen/Transports
    // haben kann, die von da abfahren, also das dann komplett rauswerfen wäre doof.
    return weight_count;
  }

  static void filter_stations(std::vector<start>& starts, timetable const& tt) {
    printf("1 Anzahl starts vorher: %llu \n", starts.size());
    if(tt.percentage_filter_) {
      percentage_filter(starts, tt.percent_for_filter_);
    }
    if(tt.weighted_filter_) {
      weighted_filter(starts, tt, tt.line_filter_);
    }
    printf("nachher: %llu \n", starts.size());
  }

};
} // namespace nigiri::routing
