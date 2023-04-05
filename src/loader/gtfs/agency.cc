#include "nigiri/loader/gtfs/agency.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

agency_map_t read_agencies(timetable& tt, std::string_view file_content) {
  struct agency {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_timezone")> tz_name_;
  };

  return utl::line_range{utl::buf_reader{file_content}}  //
         | utl::csv<agency>()  //
         | utl::transform([&](agency const& a) {
             return cista::pair{
                 a.id_->to_str(),
                 tt.register_provider({a.id_->view(), a.name_->view()})};
           })  //
         | utl::to<agency_map_t>();
}

}  // namespace nigiri::loader::gtfs
