#pragma once

#include <cassert>
#include <type_traits>

#include "cista/containers/vector.h"
#include "cista/verify.h"
#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

template <typename T, typename Tag>
struct gpu_strong : public cista::strong<T, Tag> {
  using Base = cista::strong<T, Tag>;
  using typename Base::value_t;

#ifdef NIGIRI_CUDA
  __host__ __device__ gpu_strong() = default;
  __host__ __device__ explicit constexpr gpu_strong(T const& v) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : Base{v} {}
  __host__ __device__ explicit constexpr gpu_strong(T&& v) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : Base{std::move(v)} {}
  __host__ __device__ static constexpr gpu_strong invalid() {
    return gpu_strong{std::numeric_limits<T>::max()};
  }
#endif

};
#ifdef NIGIRI_CUDA
template <typename T, typename Tag>
__host__ __device__ inline constexpr typename gpu_strong<T, Tag>::value_t gpu_to_idx(
    gpu_strong<T, Tag> const& s) {
  return s.v_;
}
#endif

//TODO: später raus kicken was nicht brauchen
using gpu_delta_t = int16_t;
using gpu_clasz_mask_t = std::uint16_t;
using gpu_location_idx_t = gpu_strong<std::uint32_t, struct _location_idx>;
using gpu_value_type = gpu_location_idx_t::value_t;
using gpu_bitfield_idx_t = gpu_strong<std::uint32_t, struct _bitfield_idx>;
using gpu_route_idx_t = gpu_strong<std::uint32_t, struct _route_idx>;
//using gpu_section_idx_t = gpu_strong<std::uint32_t, struct _section_idx>;
//using gpu_section_db_idx_t = gpu_strong<std::uint32_t, struct _section_db_idx>;
//using gpu_trip_idx_t = gpu_strong<std::uint32_t, struct _trip_idx>;
//using gpu_trip_id_idx_t = gpu_strong<std::uint32_t, struct _trip_id_str_idx>;
using gpu_transport_idx_t = gpu_strong<std::uint32_t, struct _transport_idx>;
//using gpu_source_idx_t = gpu_strong<std::uint16_t, struct _source_idx>;
using gpu_day_idx_t = gpu_strong<std::uint16_t, struct _day_idx>;
//using gpu_timezone_idx_t = gpu_strong<std::uint16_t, struct _timezone_idx>;
using gpu_clasz_mask_t = std::uint16_t;
using gpu_profile_idx_t = std::uint8_t;
using gpu_stop_idx_t = std::uint16_t;
using i16_minutes = std::chrono::duration<std::int16_t, std::ratio<60>>;
using duration_t = i16_minutes;
using gpu_minutes_after_midnight_t = duration_t;
using gpu_days = std::chrono::duration
    <int, date::detail::ratio_multiply<std::ratio<24>, std::chrono::hours::period>>;

enum class gpu_event_type { kArr, kDep };
enum class gpu_direction { kForward, kBackward };
using gpu_i32_minutes = std::chrono::duration<int32_t, std::ratio<60>>;
using gpu_unixtime_t = std::chrono::sys_time<gpu_i32_minutes>;
struct gpu_delta{
  std::uint16_t days_ : 5;
  std::uint16_t mam_ : 11;
  bool operator== (gpu_delta const& a) const{
    return (a.days_== this->days_ && a.mam_==this->mam_);
  }
  bool operator!= (gpu_delta const& a) const{
    return !(operator==(a));
  }
};
#ifdef NIGIRI_CUDA
template <typename T>
__host__ __device__ inline gpu_delta_t gpu_clamp(T t) {
#if defined(NIGIRI_TRACING)
  if (t < std::numeric_limits<gpu_delta_t>::min()) {
    trace_upd("CLAMP {} TO {}\n", t, std::numeric_limits<gpu_delta_t>::min());
  }
  if (t > std::numeric_limits<delta_t>::max()) {
    trace_upd("CLAMP {} TO {}\n", t, std::numeric_limits<gpu_delta_t>::max());
  }
#endif
  return static_cast<gpu_delta_t>(
      std::clamp(t, static_cast<int>(std::numeric_limits<gpu_delta_t>::min()),
                 static_cast<int>(std::numeric_limits<gpu_delta_t>::max())));
}
#endif
template <gpu_direction SearchDir>
inline constexpr auto const kInvalidGpuDelta =
    SearchDir == gpu_direction::kForward ? std::numeric_limits<gpu_delta_t>::min()
                                         : std::numeric_limits<gpu_delta_t>::max();
inline gpu_unixtime_t gpu_delta_to_unix(date::sys_days const base, gpu_delta_t const d) {
  return std::chrono::time_point_cast<gpu_unixtime_t::duration>(base) +
         d * gpu_unixtime_t::duration{1};
}

namespace cista {
template <typename Key, typename DataVec, typename IndexVec>
struct basic_gpu_vecvec {
  using data_value_type = typename DataVec::value_type;
  using index_value_type = typename IndexVec::value_type;
#ifdef NIGIRI_CUDA
  struct bucket final {
    using value_type = data_value_type;
    using iterator = typename DataVec::iterator;
    using const_iterator = typename DataVec::iterator;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::add_pointer_t<value_type>;
    using reference = bucket;

    __host__ __device__ bucket(basic_gpu_vecvec* map, index_value_type const i)
        : map_{map}, i_{to_idx(i)} {}

    __host__ __device__ friend data_value_type* data(bucket b) { return &b[0]; }
    __host__ __device__ friend index_value_type size(bucket b) {
      return b.size();
    }

    __host__ __device__ data_value_type const* data() const {
      return empty() ? nullptr : &front();
    }

    template <typename T = std::decay_t<data_value_type>,
              typename = std::enable_if_t<std::is_same_v<T, char>>>
    __host__ __device__ std::string_view view() const {
      return std::string_view{begin(), size()};
    }

    __host__ __device__ value_type& front() {
      assert(!empty());
      return operator[](0);
    }

    __host__ __device__ value_type& back() {
      assert(!empty());
      return operator[](size() - 1U);
    }

    __host__ __device__ bool empty() const { return begin() == end(); }

    template <typename Args>
    __host__ __device__ void push_back(Args&& args) {
      map_->data_.insert(std::next(std::begin(map_->data_), bucket_end_idx()),
                         std::forward<Args>(args));
      for (auto i = i_ + 1; i != map_->bucket_starts_.size(); ++i) {
        ++map_->bucket_starts_[i];
      }
    }

    __host__ __device__ value_type& operator[](std::size_t const i) {
      assert(is_inside_bucket(i));
      return map_->data_[to_idx(map_->bucket_starts_[i_] + i)];
    }

    __host__ __device__ value_type const& operator[](
        std::size_t const i) const {
      assert(is_inside_bucket(i));
      return map_->data_[to_idx(map_->bucket_starts_[i_] + i)];
    }

    __host__ __device__ value_type const& at(std::size_t const i) const {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ value_type& at(std::size_t const i) {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ std::size_t size() const {
      return bucket_end_idx() - bucket_begin_idx();
    }
    __host__ __device__ iterator begin() {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ iterator end() {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ const_iterator begin() const {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ const_iterator end() const {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ friend iterator begin(bucket const& b) {
      return b.begin();
    }
    __host__ __device__ friend iterator end(bucket const& b) { return b.end(); }
    __host__ __device__ friend iterator begin(bucket& b) { return b.begin(); }
    __host__ __device__ friend iterator end(bucket& b) { return b.end(); }

    __host__ __device__ friend bool operator==(bucket const& a,
                                               bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ == b.i_;
    }
    __host__ __device__ friend bool operator!=(bucket const& a,
                                               bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ != b.i_;
    }
    __host__ __device__ bucket& operator++() {
      ++i_;
      return *this;
    }
    __host__ __device__ bucket& operator--() {
      --i_;
      return *this;
    }
    __host__ __device__ bucket operator*() const { return *this; }
    __host__ __device__ bucket& operator+=(difference_type const n) {
      i_ += n;
      return *this;
    }
    __host__ __device__ bucket& operator-=(difference_type const n) {
      i_ -= n;
      return *this;
    }
    __host__ __device__ bucket operator+(difference_type const n) const {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }
    __host__ __device__ bucket operator-(difference_type const n) const {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }
    __host__ __device__ friend difference_type operator-(bucket const& a,
                                                         bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ - b.i_;
    }

  private:
    __host__ __device__ index_value_type bucket_begin_idx() const {
      return to_idx(map_->bucket_starts_[i_]);
    }
    __host__ __device__ index_value_type bucket_end_idx() const {
      return to_idx(map_->bucket_starts_[i_ + 1U]);
    }
    __host__ __device__ bool is_inside_bucket(std::size_t const i) const {
      return bucket_begin_idx() + i < bucket_end_idx();
    }

    basic_gpu_vecvec* map_;
    index_value_type i_;
  };

  struct const_bucket final {
    using value_type = data_value_type;
    using iterator = typename DataVec::const_iterator;
    using const_iterator = typename DataVec::const_iterator;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference<value_type>;

    __host__ __device__ const_bucket(basic_gpu_vecvec const* map,
                                     index_value_type const i)
        : map_{map}, i_{to_idx(i)} {}

    __host__ __device__ friend data_value_type const* data(const_bucket b) {
      return b.data();
    }
    __host__ __device__ friend index_value_type size(const_bucket b) {
      return b.size();
    }

    __host__ __device__ data_value_type const* data() const {
      return empty() ? nullptr : &front();
    }

    template <typename T = std::decay_t<data_value_type>,
              typename = std::enable_if_t<std::is_same_v<T, char>>>
    __host__ __device__ std::string_view view() const {
      return std::string_view{begin(), size()};
    }

    __host__ __device__ value_type const& front() const {
      assert(!empty());
      return operator[](0);
    }

    __host__ __device__ value_type const& back() const {
      assert(!empty());
      return operator[](size() - 1U);
    }

    __host__ __device__ bool empty() const { return begin() == end(); }

    __host__ __device__ value_type const& at(std::size_t const i) const {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ value_type const& operator[](
        std::size_t const i) const {
      assert(is_inside_bucket(i));
      return map_->data_[map_->bucket_starts_[i_] + i];
    }

    __host__ __device__ index_value_type size() const {
      return bucket_end_idx() - bucket_begin_idx();
    }
    __host__ __device__ const_iterator begin() const {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ const_iterator end() const {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ friend const_iterator begin(const_bucket const& b) {
      return b.begin();
    }
    __host__ __device__ friend const_iterator end(const_bucket const& b) {
      return b.end();
    }

    __host__ __device__ friend bool operator==(const_bucket const& a,
                                               const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ == b.i_;
    }
    __host__ __device__ friend bool operator!=(const_bucket const& a,
                                               const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ != b.i_;
    }
    __host__ __device__ const_bucket& operator++() {
      ++i_;
      return *this;
    }
    __host__ __device__ const_bucket& operator--() {
      --i_;
      return *this;
    }
    __host__ __device__ const_bucket operator*() const { return *this; }
    __host__ __device__ const_bucket& operator+=(difference_type const n) {
      i_ += n;
      return *this;
    }
    __host__ __device__ const_bucket& operator-=(difference_type const n) {
      i_ -= n;
      return *this;
    }
    __host__ __device__ const_bucket operator+(difference_type const n) const {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }
    __host__ __device__ const_bucket operator-(difference_type const n) const {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }
    __host__ __device__ friend difference_type operator-(
        const_bucket const& a, const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ - b.i_;
    }

  private:
    __host__ __device__ std::size_t bucket_begin_idx() const {
      return to_idx(map_->bucket_starts_[i_]);
    }
    __host__ __device__ std::size_t bucket_end_idx() const {
      return to_idx(map_->bucket_starts_[i_ + 1]);
    }
    __host__ __device__ bool is_inside_bucket(std::size_t const i) const {
      return bucket_begin_idx() + i < bucket_end_idx();
    }

    std::size_t i_;
    basic_gpu_vecvec const* map_;
  };

  using value_type = bucket;
  using iterator = bucket;
  using const_iterator = const_bucket;

  __host__ __device__ bucket operator[](Key const i) {
    return {this, to_idx(i)};
  }
  __host__ __device__ const_bucket operator[](Key const i) const {
    return {this, to_idx(i)};
  }

  __host__ __device__ const_bucket at(Key const i) const {
    verify(to_idx(i) < bucket_starts_.size(),
           "basic_gpu_vecvec::at: index out of range");
    return {this, to_idx(i)};
  }

  __host__ __device__ bucket at(Key const i) {
    verify(to_idx(i) < bucket_starts_.size(),
           "basic_gpu_vecvec::at: index out of range");
    return {this, to_idx(i)};
  }

  __host__ __device__ bucket front() { return at(Key{0}); }
  __host__ __device__ bucket back() { return at(Key{size() - 1}); }

  __host__ __device__ const_bucket front() const { return at(Key{0}); }
  __host__ __device__ const_bucket back() const { return at(Key{size() - 1}); }

  __host__ __device__ index_value_type size() const {
    return empty() ? 0U : bucket_starts_.size() - 1;
  }
  __host__ __device__ bool empty() const { return bucket_starts_.empty(); }

  template <typename Container,
            typename = std::enable_if_t<std::is_convertible_v<
                decltype(*std::declval<Container>().begin()),
                data_value_type>>>
  __host__ __device__ void emplace_back(Container&& bucket) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    bucket_starts_.emplace_back(
        static_cast<index_value_type>(data_.size() + bucket.size()));
    data_.insert(std::end(data_),  //
                 std::make_move_iterator(std::begin(bucket)),
                 std::make_move_iterator(std::end(bucket)));
  }

  __host__ __device__ bucket add_back_sized(std::size_t const bucket_size) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    data_.resize(data_.size() + bucket_size);
    bucket_starts_.emplace_back(static_cast<index_value_type>(data_.size()));
    return at(Key{size() - 1U});
  }

  template <typename X>
  std::enable_if_t<std::is_convertible_v<std::decay_t<X>, data_value_type>>
      __host__ __device__ emplace_back(std::initializer_list<X>&& x) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    bucket_starts_.emplace_back(
        static_cast<index_value_type>(data_.size() + x.size()));
    data_.insert(std::end(data_),  //
                 std::make_move_iterator(std::begin(x)),
                 std::make_move_iterator(std::end(x)));
  }

  template <typename T = data_value_type,
            typename = std::enable_if_t<std::is_convertible_v<T, char const>>>
  __host__ __device__ void emplace_back(char const* s) {
    return emplace_back(std::string_view{s});
  }

  __host__ __device__ void resize(std::size_t const new_size) {
    auto const old_size = bucket_starts_.size();
    bucket_starts_.resize(
        static_cast<typename IndexVec::size_type>(new_size + 1U));
    for (auto i = old_size; i < new_size + 1U; ++i) {
      bucket_starts_[i] = data_.size();
    }
  }

  __host__ __device__ bucket begin() { return bucket{this, 0U}; }
  __host__ __device__ bucket end() { return bucket{this, size()}; }
  __host__ __device__ const_bucket begin() const {
    return const_bucket{this, 0U};
  }
  __host__ __device__ const_bucket end() const {
    return const_bucket{this, size()};
  }

  __host__ __device__ friend bucket begin(basic_gpu_vecvec& m) {
    return m.begin();
  }
  __host__ __device__ friend bucket end(basic_gpu_vecvec& m) { return m.end(); }
  __host__ __device__ friend const_bucket begin(basic_gpu_vecvec const& m) {
    return m.begin();
  }
  __host__ __device__ friend const_bucket end(basic_gpu_vecvec const& m) {
    return m.end();
  }
#endif
  DataVec data_;
  IndexVec bucket_starts_;
};
namespace raw {
template <typename K, typename V, typename SizeType = cista::base_t<K>>
using gpu_vecvec = basic_gpu_vecvec<K, vector<V>, vector<SizeType>>;
}  // namespace raw

} //namespace cista

struct gpu_transport {
  CISTA_FRIEND_COMPARABLE(gpu_transport)
  CISTA_PRINTABLE(gpu_transport, "idx", "day")
#ifdef NIGIRI_CUDA
  __host__ __device__ static gpu_transport invalid() noexcept {
    return gpu_transport{};
  }
  __host__ __device__ constexpr bool is_valid() const {
    return day_ != gpu_day_idx_t::invalid();
  }
#else
  static gpu_transport invalid() noexcept {
    return gpu_transport{};
  }
  constexpr bool is_valid() const {
    return day_ != gpu_day_idx_t::invalid();
  }
#endif
  gpu_transport_idx_t t_idx_{gpu_transport_idx_t::invalid()};
  gpu_day_idx_t day_{gpu_day_idx_t::invalid()};
};