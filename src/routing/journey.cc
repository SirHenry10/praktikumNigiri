#include "nigiri/routing/journey.h"

#include "utl/enumerate.h"
#include "utl/overloaded.h"

#include "nigiri/common/indent.h"
#include "nigiri/print_transport.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void journey::leg::print(std::ostream& out,
                         timetable const& tt,
                         rt_timetable const* rtt,
                         unsigned const n_indent,
                         bool const debug) const {
  std::visit(
      utl::overloaded{[&](run_enter_exit const& t) {
                        if (t.t_.is_rt() && rtt != nullptr) {
                          print_transport(tt, rtt, out, t.t_.rt_, t.stop_range_,
                                          n_indent, debug);
                        } else {
                          print_transport(tt, rtt, out, t.t_.t_, t.stop_range_,
                                          n_indent, debug);
                        }
                      },
                      [&](footpath const x) {
                        indent(out, n_indent);
                        out << "FOOTPATH (duration=" << x.duration().count()
                            << ")\n";
                      },
                      [&](offset const x) {
                        indent(out, n_indent);
                        out << "MUMO (id=" << static_cast<int>(x.type_)
                            << ", duration=" << x.duration().count() << ")\n";
                      }},
      uses_);
}

void journey::print(std::ostream& out,
                    timetable const& tt,
                    rt_timetable const* rtt,
                    bool const debug) const {
  if (legs_.empty()) {
    out << "no legs [start_time=" << start_time_ << ", dest_time=" << dest_time_
        << ", transfers=" << static_cast<int>(transfers_) << "\n";
    return;
  }

  if (debug) {
    out << " DURATION: " << travel_time() << " ";
  }
  out << "[" << start_time_ << ", " << dest_time_ << "]\n";
  out << "TRANSFERS: " << static_cast<int>(transfers_) << "\n";
  out << "     FROM: " << location{tt, legs_.front().from_} << " ["
      << legs_.front().dep_time_ << "]\n";
  out << "       TO: " << location{tt, legs_.back().to_} << " ["
      << legs_.back().arr_time_ << "]\n";
  for (auto const [i, l] : utl::enumerate(legs_)) {
    out << "leg " << i << ": " << location{tt, l.from_} << " [" << l.dep_time_
        << "] -> " << location{tt, l.to_} << " [" << l.arr_time_ << "]\n";
    l.print(out, tt, rtt, 1, debug);
  }
}

}  // namespace nigiri::routing
