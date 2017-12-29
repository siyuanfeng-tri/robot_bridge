#pragma once

#include <Eigen/Core>
#include <time.h>

namespace robot_bridge {

inline double get_system_time() {
  struct timespec the_tp;
  clock_gettime(CLOCK_REALTIME, &the_tp);
  return ((double)(the_tp.tv_sec)) + 1.0e-9 * the_tp.tv_nsec;
}

template <typename T> const T &clamp(const T &val, const T &lo, const T &hi) {
  DRAKE_DEMAND(lo < hi);

  if (val <= lo)
    return lo;
  else if (val >= hi)
    return hi;
  else
    return val;
}

template <typename T>
T bilinear_interp(const T x, const T y, const T x1, const T x2, const T y1,
                  const T y2, const T Q11, const T Q12, const T Q21,
                  const T Q22) {
  Eigen::Matrix<T, 2, 2> Q;
  Eigen::Matrix<T, 1, 2> X;
  Eigen::Matrix<T, 2, 1> Y;

  Q << Q11, Q12, Q21, Q22;
  X << x2 - x, x - x1;
  Y << y2 - y, y - y1;

  return (X * Q * Y)(0, 0) / (x2 - x1) / (y2 - y1);
}

}  // namespace robot_bridge
