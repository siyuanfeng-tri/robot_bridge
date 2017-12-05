#pragma once

#include "drake/common/eigen_types.h"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_schunk_wsg_status.hpp"
#include "drake/manipulation/util/trajectory_utils.h"
#include "drake/multibody/rigid_body_tree.h"

namespace Eigen {
typedef Matrix<double, 6, 1> Vector6d;
}

namespace robot_bridge {

extern const std::string kEEName;

enum class MotionStatus {
  EXECUTING = 0,
  DONE = 1,
  ERR_FORCE_SAFETY = 2,
  ERR_STUCK = 3,
  ERR_INVALID_ARG = 4,
};

class RobotState {
public:
  static constexpr double kUninitTime = -1.0;

  RobotState(const RigidBodyTree<double> *iiwa,
             const RigidBodyFrame<double> *frame_T);
  bool UpdateState(const drake::lcmt_iiwa_status &msg);
  const KinematicsCache<double> &get_cache() const { return cache_; }
  const Eigen::VectorXd &get_q() const { return q_; }
  const Eigen::VectorXd &get_v() const { return v_; }
  const Eigen::VectorXd &get_ext_trq() const { return ext_trq_; }
  const Eigen::VectorXd &get_trq() const { return trq_; }
  const Eigen::Vector6d &get_ext_wrench() const { return ext_wrench_; }
  const Eigen::Isometry3d &get_X_WT() const { return X_WT_; }
  const Eigen::Vector6d &get_V_WT() const { return V_WT_; }

  Eigen::Isometry3d get_X_WP(const RigidBodyFrame<double> &P) const;
  Eigen::Vector6d get_V_WP(const RigidBodyFrame<double> &P) const;

  double get_time() const { return time_; }
  double get_dt() const { return delta_time_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const RigidBodyTree<double> *iiwa_;
  const RigidBodyFrame<double> *frame_T_;
  KinematicsCache<double> cache_;
  double time_{kUninitTime};
  double delta_time_{0};

  Eigen::VectorXd q_{};
  Eigen::VectorXd v_{};
  Eigen::VectorXd trq_{};
  Eigen::VectorXd ext_trq_{};
  Eigen::MatrixXd J_{};

  Eigen::Isometry3d X_WT_{Eigen::Isometry3d::Identity()};
  Eigen::Vector6d V_WT_{Eigen::Vector6d::Zero()};

  // J^T * F = ext_trq_
  Eigen::Vector6d ext_wrench_{Eigen::Vector6d::Zero()};

  bool init_{false};
};

class GripperState {
public:
  static constexpr double kUninitTime = -1.0;

  void UpdateState(const drake::lcmt_schunk_wsg_status &msg) {
    time_ = static_cast<double>(msg.utime) / 1e6;
    position_ = msg.actual_position_mm / 1000.;
    velocity_ = msg.actual_speed_mm_per_s / 1000.;
    force_ = msg.actual_force;
  }

  double get_time() const { return time_; }
  double get_position() const { return position_; }
  double get_velocity() const { return velocity_; }
  double get_force() const { return force_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  double time_{kUninitTime};
  double position_{0};
  double velocity_{0};
  double force_{0};
};

Eigen::Matrix<double, 7, 1> pose_to_vec(const Eigen::Isometry3d &pose);

// Solves the IK, s.t. FK(ret, frame_T) = X_WT.
Eigen::VectorXd PointIk(const Eigen::Isometry3d &X_WT,
                        const RigidBodyFrame<double> &frame_T,
                        const Eigen::VectorXd &q_ini,
                        RigidBodyTree<double> *robot);

// When at the solution, @p frame_C's origin in world would be at
// @p camera_in_world, and @p frame_C's z axis would be pointing at
// @p target_in_world.
Eigen::VectorXd GazeIk(const Eigen::Vector3d &target_in_world,
                       const Eigen::Vector3d &camera_in_world,
                       const RigidBodyFrame<double> &frame_C,
                       const Eigen::VectorXd &q_ini,
                       RigidBodyTree<double> *robot);

// When at the solution, @p frame_C's z axis would be pointing at
// @p target_in_world, and of opposite direction of
// @p target_to_camera_in_world, and the distance between @p frame_C's origin
// and @p target_in_world is larger than min_dist.
Eigen::VectorXd GazeIk2(const Eigen::Vector3d &target_in_world,
                        const Eigen::Vector3d &target_to_camera_in_world,
                        double min_dist, double max_dist,
                        const RigidBodyFrame<double> &frame_C,
                        const Eigen::VectorXd &q_ini,
                        RigidBodyTree<double> *robot);

std::vector<Eigen::VectorXd> ComputeCalibrationConfigurations(
    const RigidBodyTree<double> &robot, const RigidBodyFrame<double> &frame_C,
    const Eigen::VectorXd &q0, const Eigen::Vector3d &p_WG, double min_dist,
    double width, double height, int num_width_pt, int num_height_pt);

std::vector<Eigen::VectorXd>
ScanAroundPoint(const RigidBodyTree<double> &robot,
                const RigidBodyFrame<double> &frame_C,
                const Eigen::Vector3d &p_WP, const Eigen::Vector3d &normal_W,
                double min_dist, double max_dist, double width, double height,
                int dw = 2, int dh = 2);

PiecewisePolynomial<double>
RetimeTrajCubic(const std::vector<Eigen::MatrixXd> &q,
                const Eigen::MatrixXd &v0, const Eigen::MatrixXd &v1,
                const Eigen::MatrixXd &v_lower, const Eigen::MatrixXd &v_upper,
                const Eigen::MatrixXd &vd_lower,
                const Eigen::MatrixXd &vd_upper);

} // namespace robot_bridge
