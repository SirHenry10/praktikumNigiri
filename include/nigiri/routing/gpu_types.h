#pragma once

#include <cassert>
#include <type_traits>

#include "cista/containers/bitset.h"
#include "cista/containers/ptr.h"
#include "cista/containers/string.h"
#include "cista/containers/vector.h"
#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"
#include "cista/verify.h"
#include "geo/latlng.h"

template <typename T, typename Tag>
struct gpu_strong {
  using value_t = T;
#ifdef NIGIRI_CUDA
  __host__ __device__ gpu_strong() = default;
  __host__ __device__ explicit gpu_strong(T const& v) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : v_{v} {}
  __host__ __device__ constexpr explicit gpu_strong(T&& v) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : v_{std::move(v)} {}
  template <typename X>
#if _MSVC_LANG >= 202002L || __cplusplus >= 202002L
    requires std::is_integral_v<std::decay_t<X>> &&
             std::is_integral_v<std::decay_t<T>>
#endif
  explicit constexpr gpu_strong(X&& x) : v_{static_cast<T>(x)} {
  }

  __host__ __device__ gpu_strong(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;
  __host__ __device__ gpu_strong& operator=(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;

  __host__ __device__ gpu_strong(gpu_strong const& o) = default;
  __host__ __device__ gpu_strong& operator=(gpu_strong const& o) = default;

  __host__ __device__ static gpu_strong invalid() {
    return gpu_strong{std::numeric_limits<T>::max()};
  }

  __host__ __device__ gpu_strong& operator++() {
    ++v_;
    return *this;
  }

  __host__ __device__ gpu_strong operator++(int) {
    auto cpy = *this;
    ++v_;
    return cpy;
  }

  __host__ __device__ gpu_strong& operator--() {
    --v_;
    return *this;
  }

  __host__ __device__ const gpu_strong operator--(int) {
    auto cpy = *this;
    --v_;
    return cpy;
  }

  __host__ __device__ gpu_strong operator+(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ + s.v_)};
  }
  __host__ __device__ gpu_strong operator-(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ - s.v_)};
  }
  __host__ __device__ gpu_strong operator*(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ * s.v_)};
  }
  __host__ __device__ gpu_strong operator/(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ / s.v_)};
  }
  __host__ __device__ gpu_strong operator+(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ + i)};
  }
  __host__ __device__ gpu_strong operator-(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ - i)};
  }
  __host__ __device__ gpu_strong operator*(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ * i)};
  }
  __host__ __device__ gpu_strong operator/(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ / i)};
  }

  __host__ __device__ gpu_strong& operator+=(T const& i) {
    v_ += i;
    return *this;
  }
  __host__ __device__ gpu_strong& operator-=(T const& i) {
    v_ -= i;
    return *this;
  }

  __host__ __device__ gpu_strong operator>>(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ >> i)};
  }
  __host__ __device__ gpu_strong operator<<(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ << i)};
  }
  __host__ __device__ gpu_strong operator>>(gpu_strong const& o) const { return v_ >> o.v_; }
  __host__ __device__ gpu_strong operator<<(gpu_strong const& o) const { return v_ << o.v_; }

  __host__ __device__ gpu_strong& operator|=(gpu_strong const& o) {
    v_ |= o.v_;
    return *this;
  }
  __host__ __device__ gpu_strong& operator&=(gpu_strong const& o) {
    v_ &= o.v_;
    return *this;
  }

  __host__ __device__ bool operator==(gpu_strong const& o) const { return v_ == o.v_; }
  __host__ __device__ bool operator!=(gpu_strong const& o) const { return v_ != o.v_; }
  __host__ __device__ bool operator<=(gpu_strong const& o) const { return v_ <= o.v_; }
  __host__ __device__ bool operator>=(gpu_strong const& o) const { return v_ >= o.v_; }
  __host__ __device__ bool operator<(gpu_strong const& o) const { return v_ < o.v_; }
  __host__ __device__ bool operator>(gpu_strong const& o) const { return v_ > o.v_; }

  __host__ __device__ bool operator==(T const& o) const { return v_ == o; }
  __host__ __device__ bool operator!=(T const& o) const { return v_ != o; }
  __host__ __device__ bool operator<=(T const& o) const { return v_ <= o; }
  __host__ __device__ bool operator>=(T const& o) const { return v_ >= o; }
  __host__ __device__ bool operator<(T const& o) const { return v_ < o; }
  __host__ __device__ bool operator>(T const& o) const { return v_ > o; }

  __host__ __device__ explicit operator T const&() const& noexcept { return v_; }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& o, gpu_strong const& t) {
    return o << t.v_;
  }

#else
  constexpr gpu_strong() = default;
  explicit constexpr gpu_strong(T const& v) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : v_{v} {}
  explicit constexpr gpu_strong(T&& v) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : v_{std::move(v)} {}
  template <typename X>
#if _MSVC_LANG >= 202002L || __cplusplus >= 202002L
    requires std::is_integral_v<std::decay_t<X>> &&
             std::is_integral_v<std::decay_t<T>>
#endif
  explicit constexpr gpu_strong(X&& x) : v_{static_cast<T>(x)} {
  }

  constexpr gpu_strong(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;
  constexpr gpu_strong& operator=(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;

  constexpr gpu_strong(gpu_strong const& o) = default;
  constexpr gpu_strong& operator=(gpu_strong const& o) = default;

  static constexpr gpu_strong invalid() {
    return gpu_strong{std::numeric_limits<T>::max()};
  }

  constexpr gpu_strong& operator++() {
    ++v_;
    return *this;
  }

  constexpr gpu_strong operator++(int) {
    auto cpy = *this;
    ++v_;
    return cpy;
  }

  constexpr gpu_strong& operator--() {
    --v_;
    return *this;
  }

  constexpr const gpu_strong operator--(int) {
    auto cpy = *this;
    --v_;
    return cpy;
  }

  constexpr gpu_strong operator+(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ + s.v_)};
  }
  constexpr gpu_strong operator-(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ - s.v_)};
  }
  constexpr gpu_strong operator*(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ * s.v_)};
  }
  constexpr gpu_strong operator/(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ / s.v_)};
  }
  constexpr gpu_strong operator+(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ + i)};
  }
  constexpr gpu_strong operator-(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ - i)};
  }
  constexpr gpu_strong operator*(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ * i)};
  }
  constexpr gpu_strong operator/(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ / i)};
  }

  constexpr gpu_strong& operator+=(T const& i) {
    v_ += i;
    return *this;
  }
  constexpr gpu_strong& operator-=(T const& i) {
    v_ -= i;
    return *this;
  }

  constexpr gpu_strong operator>>(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ >> i)};
  }
  constexpr gpu_strong operator<<(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ << i)};
  }
  constexpr gpu_strong operator>>(gpu_strong const& o) const { return v_ >> o.v_; }
  constexpr gpu_strong operator<<(gpu_strong const& o) const { return v_ << o.v_; }

  constexpr gpu_strong& operator|=(gpu_strong const& o) {
    v_ |= o.v_;
    return *this;
  }
  constexpr gpu_strong& operator&=(gpu_strong const& o) {
    v_ &= o.v_;
    return *this;
  }

  constexpr bool operator==(gpu_strong const& o) const { return v_ == o.v_; }
  constexpr bool operator!=(gpu_strong const& o) const { return v_ != o.v_; }
  constexpr bool operator<=(gpu_strong const& o) const { return v_ <= o.v_; }
  constexpr bool operator>=(gpu_strong const& o) const { return v_ >= o.v_; }
  constexpr bool operator<(gpu_strong const& o) const { return v_ < o.v_; }
  constexpr bool operator>(gpu_strong const& o) const { return v_ > o.v_; }

  constexpr bool operator==(T const& o) const { return v_ == o; }
  constexpr bool operator!=(T const& o) const { return v_ != o; }
  constexpr bool operator<=(T const& o) const { return v_ <= o; }
  constexpr bool operator>=(T const& o) const { return v_ >= o; }
  constexpr bool operator<(T const& o) const { return v_ < o; }
  constexpr bool operator>(T const& o) const { return v_ > o; }

  explicit operator T const&() const& noexcept { return v_; }

  friend std::ostream& operator<<(std::ostream& o, gpu_strong const& t) {
    return o << t.v_;
  }
#endif
  T v_;
};

template <typename T>
struct gpu_base_type {
  using type = T;
};
template <typename T, typename Tag>
struct gpu_base_type<gpu_strong<T, Tag>> {
  using type = T;
};
template <typename T>
using gpu_base_t = typename gpu_base_type<T>::type;

#ifdef NIGIRI_CUDA
template <typename T, typename Tag>
__host__ __device__ inline constexpr typename gpu_strong<T, Tag>::value_t gpu_to_idx(
    gpu_strong<T, Tag> const& s) {
  return s.v_;
}
#else
template <typename T, typename Tag>
inline constexpr typename gpu_strong<T, Tag>::value_t gpu_to_idx(
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
using gpu_source_idx_t = gpu_strong<std::uint16_t, struct _source_idx>;
using gpu_day_idx_t = gpu_strong<std::uint16_t, struct _day_idx>;
template <size_t Size>
using bitset = cista::bitset<Size>;
constexpr auto const kMaxDays = 512;
using gpu_bitfield = bitset<kMaxDays>;
using gpu_timezone_idx_t = gpu_strong<std::uint16_t, struct _timezone_idx>;
using gpu_clasz_mask_t = std::uint16_t;
using gpu_profile_idx_t = std::uint8_t;
using gpu_stop_idx_t = std::uint16_t;
using i16_minutes = std::chrono::duration<std::int16_t, std::ratio<60>>;
using gpu_duration_t = i16_minutes;
using gpu_minutes_after_midnight_t = gpu_duration_t;

enum class gpu_clasz : std::uint8_t {
  kAir = 0,
  kHighSpeed = 1,
  kLongDistance = 2,
  kCoach = 3,
  kNight = 4,
  kRegionalFast = 5,
  kRegional = 6,
  kMetro = 7,
  kSubway = 8,
  kTram = 9,
  kBus = 10,
  kShip = 11,
  kOther = 12,
  kNumClasses
};

template <typename R1, typename R2>
using gpu_ratio_multiply = decltype(std::ratio_multiply<R1, R2>{});
using gpu_days = std::chrono::duration<int, gpu_ratio_multiply<std::ratio<24>, std::chrono::hours::period>>;
enum class gpu_event_type { kArr, kDep };
enum class gpu_direction { kForward, kBackward };

template <typename T>
using ptr = T*;
using gpu_string = cista::basic_string<ptr<char const>>;;
struct gpu_location_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(gpu_location_id, "id", "src")
  gpu_string id_;
  gpu_source_idx_t src_;
};
using gpu_i32_minutes = std::chrono::duration<int32_t, std::ratio<60>>;
using gpu_u8_minutes = std::chrono::duration<std::uint8_t, std::ratio<60>>;
using gpu_unixtime_t = std::chrono::sys_time<gpu_i32_minutes>;
template <class Duration>
using gpu_sys_time = std::chrono::time_point<std::chrono::system_clock, Duration>;
using gpu_sys_days    = gpu_sys_time<gpu_days>;
struct gpu_delta{
  std::uint16_t days_ : 5;
  std::uint16_t mam_ : 11;
  bool operator== (gpu_delta const& a) const{
    return (a.days_== this->days_ && a.mam_==this->mam_);
  }
  bool operator!= (gpu_delta const& a) const{
    return !(operator==(a));
  }
#ifdef NIGIRI_CUDA
  __host__ __device__ std::int16_t count() const { return days_ * 1440U + mam_; }
#endif
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

__host__ __device__ inline gpu_delta_t unix_to_gpu_delta(gpu_sys_days const base, gpu_unixtime_t const t) {
  return gpu_clamp(
      (t - std::chrono::time_point_cast<gpu_unixtime_t::duration>(base)).count());
}
#endif
template <gpu_direction SearchDir>
inline constexpr auto const kInvalidGpuDelta =
    SearchDir == gpu_direction::kForward ? std::numeric_limits<gpu_delta_t>::min()
                                         : std::numeric_limits<gpu_delta_t>::max();
inline gpu_unixtime_t gpu_delta_to_unix(gpu_sys_days const base, gpu_delta_t const d) {
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
  __host__ __device__ bool is_valid() const {
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


namespace cista {

template <typename T, template <typename> typename Ptr,
          bool IndexPointers = false, typename TemplateSizeType = std::uint32_t,
          class Allocator = allocator<T, Ptr>>
struct gpu_basic_vector {
  using size_type = gpu_base_t<TemplateSizeType>;
  using difference_type = std::ptrdiff_t;
  using access_type = TemplateSizeType;
  using reference = T&;
  using const_reference = T const&;
  using pointer = Ptr<T>;
  using const_pointer = Ptr<T const>;
  using value_type = T;
  using iterator = T*;
  using const_iterator = T const*;
  using allocator_type = Allocator;
#ifdef NIGIRI_CUDA
  __host__ __device__ explicit gpu_basic_vector(allocator_type const&) noexcept {}
  __host__ __device__ gpu_basic_vector() noexcept = default;

  __host__ __device__ explicit gpu_basic_vector(size_type const size, T init = T{},
                        Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    resize(size, std::move(init));
  }

  __host__ __device__ gpu_basic_vector(std::initializer_list<T> init,
               Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    set(init.begin(), init.end());
  }

  template <typename It>
  __host__ __device__ gpu_basic_vector(It begin_it, It end_it) {
    set(begin_it, end_it);
  }

  __host__ __device__ gpu_basic_vector(gpu_basic_vector&& o, Allocator const& alloc = Allocator{}) noexcept
      : el_(o.el_),
        used_size_(o.used_size_),
        allocated_size_(o.allocated_size_),
        self_allocated_(o.self_allocated_) {
    CISTA_UNUSED_PARAM(alloc)
    o.reset();
  }

  __host__ __device__ gpu_basic_vector(gpu_basic_vector const& o, Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    set(o);
  }

  __host__ __device__ gpu_basic_vector& operator=(gpu_basic_vector&& arr) noexcept {
    deallocate();

    el_ = arr.el_;
    used_size_ = arr.used_size_;
    self_allocated_ = arr.self_allocated_;
    allocated_size_ = arr.allocated_size_;

    arr.reset();
    return *this;
  }

  __host__ __device__ gpu_basic_vector& operator=(gpu_basic_vector const& arr) {
    if (&arr != this) {
      set(arr);
    }
    return *this;
  }

  __host__ __device__ ~gpu_basic_vector() { deallocate(); }

  __host__ __device__ void deallocate() {
    if (!self_allocated_ || el_ == nullptr) {
      return;
    }

    for (auto& el : *this) {
      el.~T();
    }

    std::free(el_);  // NOLINT
    reset();
  }

  __host__ __device__ allocator_type get_allocator() const noexcept { return {}; }

  __host__ __device__ T const* data() const noexcept { return begin(); }
  __host__ __device__ T* data() noexcept { return begin(); }
  __host__ __device__ T const* begin() const noexcept { return el_; }
  __host__ __device__ T const* end() const noexcept { return el_ + used_size_; }  // NOLINT
  __host__ __device__ T const* cbegin() const noexcept { return el_; }
  __host__ __device__ T const* cend() const noexcept { return el_ + used_size_; }  // NOLINT
  __host__ __device__ T* begin() noexcept { return el_; }
  __host__ __device__ T* end() noexcept { return el_ + used_size_; }  // NOLINT

  __host__ __device__ std::reverse_iterator<T const*> rbegin() const {
    return std::reverse_iterator<T*>(el_ + size());  // NOLINT
  }
  __host__ __device__ std::reverse_iterator<T const*> rend() const {
    return std::reverse_iterator<T*>(el_);
  }
  __host__ __device__ std::reverse_iterator<T*> rbegin() {
    return std::reverse_iterator<T*>(el_ + size());  // NOLINT
  }
  __host__ __device__ std::reverse_iterator<T*> rend() { return std::reverse_iterator<T*>(el_); }

  __host__ __device__ friend T const* begin(gpu_basic_vector const& a) noexcept { return a.begin(); }
  __host__ __device__ friend T const* end(gpu_basic_vector const& a) noexcept { return a.end(); }

  __host__ __device__ friend T* begin(gpu_basic_vector& a) noexcept { return a.begin(); }
  __host__ __device__ friend T* end(gpu_basic_vector& a) noexcept { return a.end(); }

  __host__ __device__ T const& operator[](access_type const index) const noexcept {
    assert(el_ != nullptr && index < used_size_);
    return el_[gpu_to_idx(index)];
  }
  __host__ __device__ T& operator[](access_type const index) noexcept {
    assert(el_ != nullptr && index < used_size_);
    return el_[gpu_to_idx(index)];
  }

  T& at(access_type const index) {
    if (index >= used_size_) {
      throw std::out_of_range{"vector::at(): invalid index"};
    }
    return (*this)[index];
  }

  __host__ __device__ T const& at(access_type const index) const {
    return const_cast<gpu_basic_vector*>(this)->at(index);
  }

  __host__ __device__ T const& back() const noexcept { return ptr_cast(el_)[used_size_ - 1]; }
  __host__ __device__ T& back() noexcept { return ptr_cast(el_)[used_size_ - 1]; }

  __host__ __device__ T& front() noexcept { return ptr_cast(el_)[0]; }
  __host__ __device__ T const& front() const noexcept { return ptr_cast(el_)[0]; }

  __host__ __device__ size_type size() const noexcept { return used_size_; }
  __host__ __device__ bool empty() const noexcept { return size() == 0U; }

  template <typename It>
  __host__ __device__ void set(It begin_it, It end_it) {
    auto const range_size = std::distance(begin_it, end_it);
#ifndef __CUDA_ARCH__
    verify(
        range_size >= 0 && range_size <= std::numeric_limits<size_type>::max(),
        "cista::vector::set: invalid range");
#endif
    reserve(static_cast<size_type>(range_size));

    auto copy_source = begin_it;
    auto copy_target = el_;
    for (; copy_source != end_it; ++copy_source, ++copy_target) {
      new (copy_target) T{std::forward<decltype(*copy_source)>(*copy_source)};
    }

    used_size_ = static_cast<size_type>(range_size);
  }

  __host__ __device__ void set(gpu_basic_vector const& arr) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (arr.used_size_ != 0U) {
        reserve(arr.used_size_);
        #ifdef __CUDA_ARCH__
          cudaMemcpy(data(), arr.data(), arr.used_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        #else
          std::memcpy(data(), arr.data(), arr.used_size_ * sizeof(T));
        #endif
      }
      used_size_ = arr.used_size_;
    } else {
      set(std::begin(arr), std::end(arr));
    }
  }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& out, gpu_basic_vector const& v) {
    out << "[\n  ";
    auto first = true;
    for (auto const& e : v) {
      if (!first) {
        out << ",\n  ";
      }
      out << e;
      first = false;
    }
    return out << "\n]";
  }


  template <class It>
  __host__ __device__ T* insert(T* pos, It first, It last) {
    return insert(pos, first, last,
                  typename std::iterator_traits<It>::iterator_category());
  }

  __host__ __device__ void pop_back() noexcept(noexcept(std::declval<T>().~T())) {
    --used_size_;
    el_[used_size_].~T();
  }

  __host__ __device__ void clear() {
    for (auto& el : *this) {
      el.~T();
    }
    used_size_ = 0;
  }
  template <typename TemplateSizeType>
  __host__ __device__ TemplateSizeType gpu_next_power_of_two(TemplateSizeType n) noexcept {
    --n;
    n |= n >> 1U;
    n |= n >> 2U;
    n |= n >> 4U;
    n |= n >> 8U;
    n |= n >> 16U;
    if constexpr (sizeof(TemplateSizeType) > 32U) {
      n |= n >> 32U;
    }
    ++n;
    return n;
  }

  __host__ __device__ void reserve(size_type new_size) {
    new_size = std::max(allocated_size_, new_size);

    if (allocated_size_ >= new_size) {
      return;
    }

    auto next_size = gpu_next_power_of_two(new_size);
    auto num_bytes = static_cast<std::size_t>(next_size) * sizeof(T);
    // GPU-spezifische Speicherallokation
    T* mem_buf;
#ifdef __CUDA_ARCH__
    cudaError_t err = cudaMalloc(&mem_buf, num_bytes);
    if (err != cudaSuccess) {
      printf("GPU allocation failed\n");
      return;
    }
#else
    mem_buf = static_cast<T*>(std::malloc(num_bytes));  // NOLINT
    if (mem_buf == nullptr) {
      throw std::bad_alloc();
    }
#endif
    if (size() != 0) {
      T* move_target = mem_buf;
      for (auto& el : *this) {
        new (move_target++) T(std::move(el));
        el.~T();
      }
    }

    T* free_me = el_;
    el_ = mem_buf;

    if (self_allocated_) {
    #ifdef __CUDA_ARCH__
          cudaFree(free_me);
    #else
          std::free(free_me);  // NOLINT
    #endif
    }

    self_allocated_ = true;
    allocated_size_ = next_size;
  }

  __host__ __device__ T* erase(T* first, T* last) {
    if (first != last) {
      auto const new_end = std::move(last, end(), first);
      for (auto it = new_end; it != end(); ++it) {
        it->~T();
      }
      used_size_ -= static_cast<size_type>(std::distance(new_end, end()));
    }
    return end();
  }

  __host__ __device__ bool contains(T const* el) const noexcept {
    return el >= begin() && el < end();
  }

  __host__ __device__ std::size_t index_of(T const* el) const noexcept {
    assert(contains(el));
    return std::distance(begin(), el);
  }

  __host__ __device__ friend bool operator==(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }
  __host__ __device__ friend bool operator!=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a == b);
  }
  __host__ __device__ friend bool operator<(gpu_basic_vector const& a, gpu_basic_vector const& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  }
  __host__ __device__ friend bool operator>(gpu_basic_vector const& a, gpu_basic_vector const& b) noexcept {
    return b < a;
  }
  __host__ __device__ friend bool operator<=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a > b);
  }
  __host__ __device__ friend bool operator>=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a < b);
  }

  __host__ __device__ void reset() noexcept {
    el_ = nullptr;
    used_size_ = {};
    allocated_size_ = {};
    self_allocated_ = false;
  }
#endif
  Ptr<T> el_{nullptr};
  size_type used_size_{0U};
  size_type allocated_size_{0U};
  bool self_allocated_{false};
  std::uint8_t __fill_0__{0U};
  std::uint16_t __fill_1__{0U};
  std::uint32_t __fill_2__{0U};
};

namespace raw {

template <typename Key, typename Value>
using gpu_vector_map = gpu_basic_vector<Value, ptr, false, Key>;

}  // namespace raw
#undef CISTA_TO_VEC

}  // namespace cista

template <typename K, typename V>
using gpu_vector_map = cista::raw::gpu_vector_map<K, V>;

namespace nigiri {

template <typename T>
struct gpu_interval {
#ifdef NIGIRI_CUDA
  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = value_type;
    CISTA_FRIEND_COMPARABLE(iterator);
    __host__ __device__ value_type operator*() const { return t_; }
    __host__ __device__ iterator& operator++() {
      ++t_;
      return *this;
    }
    __host__ __device__ iterator& operator--() {
      --t_;
      return *this;
    }
    __host__ __device__ iterator& operator+=(difference_type const x) {
      t_ += x;
      return *this;
    }
    __host__ __device__ iterator& operator-=(difference_type const x) {
      t_ -= x;
      return *this;
    }
    __host__ __device__ iterator operator+(difference_type const x) const { return *this += x; }
    __host__ __device__ iterator operator-(difference_type const x) const { return *this -= x; }
    __host__ __device__ friend difference_type operator-(iterator const& a, iterator const& b) {
      return static_cast<difference_type>(cista::to_idx(a.t_) -
                                          cista::to_idx(b.t_));
    }
    T t_;
  };

  template <typename X>
  __host__ __device__ gpu_interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  __host__ __device__ gpu_interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  __host__ __device__ operator gpu_interval<X>() {
    return {from_, to_};
  }

  __host__ __device__ T clamp(T const x) const { return std::clamp(x, from_, to_); }

  __host__ __device__ bool contains(T const t) const { return t >= from_ && t < to_; }

  __host__ __device__ bool overlaps(gpu_interval const& o) const {
    return from_ < o.to_ && to_ > o.from_;
  }

  __host__ __device__ iterator begin() const { return {from_}; }
  __host__ __device__ iterator end() const { return {to_}; }
  __host__ __device__ friend iterator begin(gpu_interval const& r) { return r.begin(); }
  __host__ __device__ friend iterator end(gpu_interval const& r) { return r.end(); }

  __host__ __device__ std::reverse_iterator<iterator> rbegin() const {
    return std::reverse_iterator<iterator>{iterator{to_}};
  }
  __host__ __device__ std::reverse_iterator<iterator> rend() const {
    return std::reverse_iterator<iterator>{iterator{from_}};
  }
  __host__ __device__ friend std::reverse_iterator<iterator> rbegin(gpu_interval const& r) {
    return r.begin();
  }
  __host__ __device__ friend std::reverse_iterator<iterator> rend(gpu_interval const& r) {
    return r.end();
  }

  __host__ __device__ auto size() const { return to_ - from_; }

  __host__ __device__ T operator[](std::size_t const i) const {
    assert(contains(from_ + static_cast<T>(i)));
    return from_ + static_cast<T>(i);
  }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& out, gpu_interval const& i) {
    return out << "[" << i.from_ << ", " << i.to_ << "[";
  }
#else
  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = value_type;
    CISTA_FRIEND_COMPARABLE(iterator);
    value_type operator*() const { return t_; }
    iterator& operator++() {
      ++t_;
      return *this;
    }
    iterator& operator--() {
      --t_;
      return *this;
    }
    iterator& operator+=(difference_type const x) {
      t_ += x;
      return *this;
    }
    iterator& operator-=(difference_type const x) {
      t_ -= x;
      return *this;
    }
    iterator operator+(difference_type const x) const { return *this += x; }
    iterator operator-(difference_type const x) const { return *this -= x; }
    friend difference_type operator-(iterator const& a, iterator const& b) {
      return static_cast<difference_type>(cista::to_idx(a.t_) -
                                          cista::to_idx(b.t_));
    }
    T t_;
  };

  template <typename X>
  gpu_interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  gpu_interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  operator gpu_interval<X>() {
    return {from_, to_};
  }

  T clamp(T const x) const { return std::clamp(x, from_, to_); }

  bool contains(T const t) const { return t >= from_ && t < to_; }

  bool overlaps(gpu_interval const& o) const {
    return from_ < o.to_ && to_ > o.from_;
  }

  iterator begin() const { return {from_}; }
  iterator end() const { return {to_}; }
  friend iterator begin(gpu_interval const& r) { return r.begin(); }
  friend iterator end(gpu_interval const& r) { return r.end(); }

  std::reverse_iterator<iterator> rbegin() const {
    return std::reverse_iterator<iterator>{iterator{to_}};
  }
  std::reverse_iterator<iterator> rend() const {
    return std::reverse_iterator<iterator>{iterator{from_}};
  }
  friend std::reverse_iterator<iterator> rbegin(gpu_interval const& r) {
    return r.begin();
  }
  friend std::reverse_iterator<iterator> rend(gpu_interval const& r) {
    return r.end();
  }

  auto size() const { return to_ - from_; }

  T operator[](std::size_t const i) const {
    assert(contains(from_ + static_cast<T>(i)));
    return from_ + static_cast<T>(i);
  }

  friend std::ostream& operator<<(std::ostream& out, gpu_interval const& i) {
    return out << "[" << i.from_ << ", " << i.to_ << "[";
  }
#endif
  T from_{}, to_{};
};

template <typename T, typename T1, typename = std::common_type_t<T1, T>>
gpu_interval(T, T1) -> gpu_interval<std::common_type_t<T, T1>>;

}  // namespace nigiri

template <typename T>
using gpu_interval = nigiri::gpu_interval<T>;

template <typename T>
struct fmt::formatter<nigiri::gpu_interval<T>> : ostream_formatter {};

template <typename T>
using gpu_interval = nigiri::gpu_interval<T>;


template <typename K, typename V, typename SizeType = gpu_base_t<K>>
using gpu_vecvec = cista::raw::gpu_vecvec<K, V, SizeType>;

template <typename V, std::size_t SIZE>
using array = cista::raw::array<V, SIZE>;
namespace nigiri{

struct gpu_footpath {
  using value_type = gpu_location_idx_t::value_t;
  static constexpr auto const kTotalBits = 8 * sizeof(value_type);
  static constexpr auto const kTargetBits = 22U;
  static constexpr auto const kDurationBits = kTotalBits - kTargetBits;
  static constexpr auto const kMaxDuration = gpu_duration_t{
      std::numeric_limits<gpu_location_idx_t::value_t>::max() >> kTargetBits};
  
  gpu_footpath() = default;

  gpu_footpath(gpu_location_idx_t::value_t const val) {
    std::memcpy(this, &val, sizeof(value_type));
  }

  gpu_footpath(gpu_location_idx_t const target, gpu_duration_t const duration)
      : target_{target},
        duration_{static_cast<value_type>(
            (duration > kMaxDuration ? kMaxDuration : duration).count())} {
  }

  gpu_location_idx_t target() const { return gpu_location_idx_t{target_}; }
  gpu_duration_t duration() const { return gpu_duration_t{duration_}; }

  gpu_location_idx_t::value_t value() const {
    return *reinterpret_cast<gpu_location_idx_t::value_t const*>(this);
  }

  friend std::ostream& operator<<(std::ostream& out, gpu_footpath const& fp) {
    return out << "(" << fp.target() << ", " << fp.duration() << ")";
  }

  friend bool operator==(gpu_footpath const& a, gpu_footpath const& b) {
    return a.value() == b.value();
  }

  friend bool operator<(gpu_footpath const& a, gpu_footpath const& b) {
    return a.value() < b.value();
  }

  gpu_location_idx_t::value_t target_ : kTargetBits;
  gpu_location_idx_t::value_t duration_ : kDurationBits;
};

template <typename Ctx>
inline void serialize(Ctx&, gpu_footpath const*, cista::offset_t const) {}

template <typename Ctx>
inline void deserialize(Ctx const&, gpu_footpath*) {}

struct gpu_locations_device {
  gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes> transfer_time_;
  gpu_vecvec<gpu_location_idx_t, gpu_footpath> gpu_footpaths_out_; //nigiri::kMaxProfiles is the size
  gpu_vecvec<gpu_location_idx_t, gpu_footpath> gpu_footpaths_in_;  //same here
};
}//namespace: nigiri
using gpu_locations = nigiri::gpu_locations_device;

constexpr gpu_duration_t operator""_gpu_days(unsigned long long n) {
  return gpu_duration_t{n * 1440U};
}
static constexpr auto const gpu_kMaxTransfers = std::uint8_t{7U};
static constexpr auto const gpu_kMaxTravelTime = 1_gpu_days;

enum class gpu_special_station : gpu_location_idx_t::value_t {
  kStart,
  kEnd,
  kVia0,
  kVia1,
  kVia2,
  kVia3,
  kVia4,
  kVia5,
  kVia6,
  kSpecialStationsSize
};

constexpr bool is_special(gpu_location_idx_t const l) {
  constexpr auto const max =
      static_cast<std::underlying_type_t<gpu_special_station>>(
          gpu_special_station::kSpecialStationsSize);
  return gpu_to_idx(l) < max;
}

constexpr auto const gpu_special_stations_names =
    cista::array<std::string_view,
                 static_cast<std::underlying_type_t<gpu_special_station>>(
                     gpu_special_station::kSpecialStationsSize)>{
        "START", "END", "VIA0", "VIA1", "VIA2", "VIA3", "VIA4", "VIA5", "VIA6"};

constexpr gpu_location_idx_t get_gpu_special_station(gpu_special_station const x) {
  return gpu_location_idx_t{
      static_cast<std::underlying_type_t<gpu_special_station>>(x)};
}

constexpr std::string_view get_gpu_special_station_name(gpu_special_station const x) {
  return gpu_special_stations_names
      [static_cast<std::underlying_type_t<gpu_special_station>>(x)];
}