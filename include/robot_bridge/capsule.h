#pragma once

#include "drake/multibody/rigid_body_tree.h"

namespace robot_bridge {

class Capsule {
public:
  Capsule(const RigidBodyFrame<double> &X_BC, double length, double radius);

  Eigen::Vector3d GetClosestPointOnAxis(const Eigen::Vector3d &pt) const;
  Eigen::Matrix3d ComputeEscapeFrame(const Eigen::Vector3d &closest,
                                     const Eigen::Vector3d &point) const;

  double GetClosestPointsOnAxis(const Capsule &other, Eigen::Vector3d *my_point,
                                Eigen::Vector3d *other_point) const;

  const RigidBody<double> &get_body() const { return X_BC_.get_rigid_body(); }

  const RigidBodyFrame<double> &get_frame() const { return X_BC_; }

  double get_radius() const { return radius_; }

  double get_length() const { return length_; }

  void set_pose(const Eigen::Isometry3d &pose) { X_WC_ = pose; }

  const Eigen::Isometry3d &get_pose() const { return X_WC_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // transformation to the attached body frame.
  // C is defined to be at the center of the capsule. Z axis is the axis.
  const RigidBodyFrame<double> X_BC_;

  Eigen::Isometry3d X_WC_{Eigen::Isometry3d::Identity()};

  double length_{};
  double radius_{};
};

} // namespace robot_bridge
