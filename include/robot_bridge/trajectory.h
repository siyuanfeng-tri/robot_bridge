#pragma once

#include "drake/manipulation/util/trajectory_utils.h"

namespace drake {
namespace manipulation {

template <typename T>
class SingleSegmentCartesianTrajectory {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SingleSegmentCartesianTrajectory)

  SingleSegmentCartesianTrajectory() {}

  /**
   * Constructor.
   * @param pos_traj Position trajectory.
   * @param rot_traj Orientation trajectory.
   */
  SingleSegmentCartesianTrajectory(
      const Isometry3<T>& X0, const Isometry3<T>& X1,
      const Vector3<T>& vel0, const Vector3<T>& vel1,
      double w0, double w1,
      double t0, double t1) {
    std::vector<MatrixX<T>> pos_knots = {X0.translation(), X1.translation()};
    position_ = PiecewiseCubicTrajectory<T>(
        PiecewisePolynomial<T>::Cubic({t0, t1}, pos_knots, vel0, vel1));

    // X1 = X0 * diff; diff = X0.inv() * X1;
    AngleAxis<T> diff(X0.linear().transpose() * X1.linear());
    axis_ = diff.axis();
    std::vector<MatrixX<T>> ang_knots = {Vector1<T>(0), Vector1<T>(diff.angle())};
    angle_ = PiecewiseCubicTrajectory<T>(
        PiecewisePolynomial<T>::Cubic({t0, t1}, ang_knots, Vector1<T>(w0), Vector1<T>(w1)));
    rot0_ = X0.linear();
  }

  /**
   * Returns the interpolated pose at @p time.
   */
  Isometry3<T> get_pose(double time) const {
    Isometry3<T> pose = Isometry3<T>::Identity();
    pose.translation() = position_.get_position(time);
    pose.linear() = rot0_ * AngleAxis<T>(angle_.get_position(time)(0, 0), axis_).toRotationMatrix();
    return pose;
  }

  /**
   * Returns the interpolated velocity at @p time or zero if @p time is before
   * this trajectory's start time or after its end time.
   */
  Vector6<T> get_velocity(double time) const {
    Vector6<T> velocity;
    velocity.template head<3>() = angle_.get_velocity(time)(0, 0) * rot0_ * axis_;
    velocity.template tail<3>() = position_.get_velocity(time);
    return velocity;
  }

  /**
   * Returns the interpolated acceleration at @p time or zero if @p time is
   * before this trajectory's start time or after its end time.
   */
  Vector6<T> get_acceleration(double time) const {
    Vector6<T> acceleration;
    acceleration.template head<3>() = angle_.get_acceleration(time)(0, 0) * rot0_ * axis_;
    acceleration.template tail<3>() = position_.get_acceleration(time);
    return acceleration;
  }

  /**
   * Returns true if the position and orientation trajectories are both
   * within @p tol from the other's.
   */
  bool is_approx(const SingleSegmentCartesianTrajectory<T>& other,
                 const T& tol) const {
    bool ret = position_.is_approx(other.position_, tol);
    ret &= axis_.isApprox(other.axis_, tol);
    ret &= angle_.is_approx(other.angle_, tol);
    ret &= rot0_.isApprox(other.rot0_, tol);
    return ret;
  }

  /**
   * Returns the position trajectory.
   */
  const PiecewiseCubicTrajectory<T>& get_position_trajectory() const {
    return position_;
  }

  /**
   * Returns the orientation trajectory.
   */
  const PiecewiseCubicTrajectory<T>& get_angle_trajectory() const {
    return angle_;
  }

  const Vector3<T>& get_axis() const {
    return axis_;
  }

 private:
  PiecewiseCubicTrajectory<T> position_;
  // X1 = X0 * diff; diff = X0.inv() * X1;
  PiecewiseCubicTrajectory<T> angle_;
  Vector3<T> axis_;
  Matrix3<T> rot0_;
};

}  // namespace manipulation
}  // namespace drake
