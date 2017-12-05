#include "robot_bridge/capsule.h"
#include "robot_bridge/util.h"

#include "drake/multibody/rigid_body_tree.h"

namespace robot_bridge {

Capsule::Capsule(const RigidBodyFrame<double>& X_BC,
                 double length,
                 double radius)
    : X_BC_(X_BC), length_(length), radius_(radius) {
}

Eigen::Vector3d Capsule::GetClosestPointOnAxis(
    const Eigen::Vector3d& pt) const {
  Eigen::Vector3d mid2point = pt - X_WC_.translation();
  Eigen::Vector3d dir = X_WC_.linear().col(2);
  double proj_len = mid2point.dot(dir);
  proj_len = clamp(proj_len, -length_ / 2., length_ / 2.);
  return dir * proj_len + X_WC_.translation();
}

Eigen::Matrix3d Capsule::ComputeEscapeFrame(
    const Eigen::Vector3d& closest,
    const Eigen::Vector3d& point) const {
  Eigen::Matrix3d R;
  Eigen::Vector3d Rx, Ry, Rz;

  Rz = (point - closest);
  if (Rz.norm() < 1e-5) {
    std::cerr << "getEscapeFrame point is on dir\n";
    Rz = X_WC_.linear().col(0);
  }
  else {
    Rz.normalize();
  }
  Rx = X_WC_.linear().col(2);
  Ry = Rz.cross(Rx).normalized();
  Rx = Ry.cross(Rz).normalized();
  R.col(0) = Rx;
  R.col(1) = Ry;
  R.col(2) = Rz;
  return R;
}

double Capsule::GetClosestPointsOnAxis(const Capsule& other,
    Eigen::Vector3d* my_point,
    Eigen::Vector3d* other_point) const {
  // http://geomalgorithms.com/a07-_distance.html
  Eigen::Vector3d u = X_WC_.linear().col(2);
  Eigen::Vector3d v = other.X_WC_.linear().col(2);
  Eigen::Vector3d P = X_WC_.translation();
  Eigen::Vector3d Q = other.X_WC_.translation();

  double a = u.dot(u);
  double b = u.dot(v);
  double c = v.dot(v);
  double d = u.dot(P - Q);
  double e = v.dot(P - Q);

  double tmp = a * c - b * b;
  // two lines are parellel
  if (std::fabs(tmp) < 1e-5) {
    *my_point = P;
    *other_point = Q;
    return (P - Q).norm();
  }

  double sc = (b * e - c * d) / tmp;
  double tc = (a * e - b * d) / tmp;

  sc = clamp(sc, -length_ / 2., length_ / 2.);
  tc = clamp(tc, -other.length_ / 2., other.length_ / 2.);

  *my_point = P + u * sc;
  *other_point = Q + v * tc;

  double d0 = (*my_point - *other_point).norm();
  Eigen::Vector3d my_point1 = GetClosestPointOnAxis(*other_point);
  Eigen::Vector3d other_point1 = other.GetClosestPointOnAxis(*my_point);
  double d1 = (my_point1 - other_point1).norm();
  double d2 = (my_point1 - other_point1).norm();

  if (d0 > d1)
    *other_point = other_point1;
  if (d0 > d2)
    *my_point = my_point1;
  return (*my_point - *other_point).norm();
}

} // namespace robot_bridge
