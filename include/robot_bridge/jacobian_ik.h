#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/trajectories/piecewise_polynomial_trajectory.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/solvers/scs_solver.h"
#include "robot_bridge/capsule.h"

namespace Eigen {
typedef Matrix<double, 6, 1> Vector6d;
}

namespace robot_bridge {

class JacobianIk {
public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JacobianIk)

  /**
   * X_W1 = X_WErr * X_W0 <=> X_WErr = X_W1 * X_W0.inv()
   * p_err = pose1.translation() - pos0.translation()
   * R_err = pose1.linear() * pose0.linear().transpose().
   */
  static Eigen::Vector6d
  ComputePoseDiffInWorldFrame(const Eigen::Isometry3d &pose0,
                              const Eigen::Isometry3d &pose1);

  /**
   * Assumes that q.size() == v.size().
   */
  JacobianIk(const RigidBodyTree<double> *robot);

  void SetJointSpeedLimit(const Eigen::VectorXd &v_upper,
                          const Eigen::VectorXd &v_lower);

  void SetJointAccelerationLimit(const Eigen::VectorXd &vd_upper,
                                 const Eigen::VectorXd &vd_lower);

  /**
   * Computes a generalized velocity that would "best" achieve the desired
   * end effector velocity subject to: generalized position, velocity and
   * constraints, collision constraints. The instantaneous direction of the
   * end effector motion is also constrained to be the same as @p V_WE.
   * The problem is setup as a QP, the cost function penalizes the magnitude
   * difference of the end effector velocity and position difference from the
   * nominal posture.
   *
   * @param cache0 The current state of the robot.
   * @param collisions A vector of pairs of collision capsules to be enforced.
   * @param frame_E The E frame
   * @param V_WE Desired end effector (frame E) velocity in the world frame.
   * @param dt Delta time.
   * @param q_nominal Nominal posture of the robot, used to resolve redundancy.
   * @param v_last Last computed v. Used to enforce acceleration constraints.
   * @param is_stuck If set to true, the QP is stuck in a local minima and
   * unable to achieve @p V_WE given other constraints.
   * @param gain_E Gain on V_WE_E specified in the E frame. Can be set to zero
   * to disable tracking.
   *
   * @return Resulting generalized velocity.
   */
  Eigen::VectorXd ComputeDofVelocity(
      const KinematicsCache<double> &cache0,
      const std::vector<std::pair<Capsule, Capsule>> &collisions,
      const RigidBodyFrame<double> &frame_E, const Eigen::Vector6d &V_WE,
      double dt, const Eigen::VectorXd &q_nominal,
      const Eigen::VectorXd &v_last, bool *is_stuck,
      const Eigen::Vector6d &gain_E = Eigen::Vector6d::Constant(1)) const;

  /**
   * Returns constant reference to the robot model.
   */
  const RigidBodyTree<double> &get_robot() const { return *robot_; }

  const Eigen::VectorXd &get_velocity_upper_limit() const { return v_upper_; }
  const Eigen::VectorXd &get_velocity_lower_limit() const { return v_lower_; }
  const Eigen::VectorXd &get_acceleration_upper_limit() const {
    return vd_upper_;
  }
  const Eigen::VectorXd &get_acceleration_lower_limit() const {
    return vd_lower_;
  }

  bool Plan(const Eigen::VectorXd &q0, const std::vector<double> &times,
            const std::vector<Eigen::Isometry3d> &pose_traj,
            const RigidBodyFrame<double> &frame_E,
            const Eigen::VectorXd &q_nominal,
            std::vector<Eigen::VectorXd> *q_sol) const;

private:
  void Setup();
  const RigidBodyTree<double> *robot_{nullptr};

  Eigen::VectorXd q_lower_;
  Eigen::VectorXd q_upper_;
  Eigen::VectorXd v_lower_;
  Eigen::VectorXd v_upper_;
  Eigen::VectorXd vd_lower_;
  Eigen::VectorXd vd_upper_;
  Eigen::VectorXd unconstrained_dof_v_limit_;
  Eigen::MatrixXd identity_;
  Eigen::VectorXd zero_;

  mutable drake::solvers::ScsSolver solver_;
};

} // namespace robot_bridge
