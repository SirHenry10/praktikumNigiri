#pragma once

#include "nigiri/routing/query.h"

namespace nigiri {
struct timetable;
struct transport;
}  // namespace nigiri

namespace nigiri::routing {

// Takes a concrete transport and query settings from an existing query object
// and generates an ontrip-train query by filling in offsets for all stops along
// the train's stop sequence starting with the given stop_idx.
query generate_ontrip_train_query(timetable const&,
                                  transport const&,
                                  unsigned const stop_idx,
                                  query const&);

}  // namespace nigiri::routing