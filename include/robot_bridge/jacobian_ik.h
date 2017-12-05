#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "robot_bridge/capsule.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/trajectories/piecewise_polynomial_trajectory.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/solvers/gurobi_solver.h"

namespace Eigen {
typedef Matrix<double, 6, 1> Vector6d;
}

namespace robot_bridge {

class JacobianIk {
public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JacobianIk)

  /**
   * Linear = pose1.translation - pos0.translation
   * Angular: R_err = pose1.linear() * pose0.linear().transpose().
   */
  static Eigen::Vector6d
  ComputePoseDiffInWorldFrame(const Eigen::Isometry3d &pose0,
                              const Eigen::Isometry3d &pose1);

  /**
   * Assumes that q.size() == v.size().
   */
  JacobianIk(const std::string &model_path,
             const Eigen::Isometry3d &base_to_world);

  JacobianIk(const RigidBodyTree<double> *robot);

  bool Plan(const Eigen::VectorXd &q0, const std::vector<double> &times,
            const std::vector<Eigen::Isometry3d> &pose_traj,
            const RigidBodyFrame<double> &frame_E,
            const Eigen::VectorXd &q_nominal,
            std::vector<Eigen::VectorXd> *q_sol) const;

  void SetJointSpeedLimit(const Eigen::VectorXd &v_upper,
                          const Eigen::VectorXd &v_lower);

  /**
   * @param cache0 Captures the current state of the robot.
   * @param V_WE Desired end effector (frame E) velocity in the world frame.
   * @param dt Delta time.
   * @param gain_E Gain on V_WE_E specified in the end effector frame.
   * @return Resulting generalized velocity.
   */
  Eigen::VectorXd ComputeDofVelocity(
      const KinematicsCache<double> &cache0,
      const std::vector<std::pair<Capsule, Capsule>>& collisions,
      const RigidBodyFrame<double> &frame_E, const Eigen::Vector6d &V_WE,
      const Eigen::VectorXd &q_nominal, double dt, bool *is_stuck,
      const Eigen::Vector6d &gain_E = Eigen::Vector6d::Constant(1)) const;

  /**
   * Returns constant reference to the robot model.
   */
  const RigidBodyTree<double> &get_robot() const { return *robot_; }

  void SetSamplingDt(double dt) { sampling_dt_ = dt; }

  double GetSamplingDt() const { return sampling_dt_; }

  const Eigen::VectorXd &get_velocity_upper_limit() const { return v_upper_; }
  const Eigen::VectorXd &get_velocity_lower_limit() const { return v_lower_; }

private:
  void Setup();
  std::unique_ptr<RigidBodyTree<double>> owned_robot_{nullptr};
  const RigidBodyTree<double> *robot_{nullptr};
  double sampling_dt_{5e-3};

  Eigen::VectorXd q_lower_;
  Eigen::VectorXd q_upper_;
  Eigen::VectorXd v_lower_;
  Eigen::VectorXd v_upper_;
  Eigen::VectorXd unconstrained_dof_v_limit_;
  Eigen::MatrixXd identity_;
  Eigen::VectorXd zero_;

  mutable drake::solvers::GurobiSolver solver_;
};

} // namespace robot_bridge
