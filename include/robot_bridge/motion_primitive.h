#pragma once

#include "robot_bridge/robot_bridge_common.h"

#include "drake/common/eigen_types.h"
#include "drake/multibody/rigid_body_tree.h"

#include "drake/manipulation/util/trajectory_utils.h"

#include "robot_bridge/jacobian_ik.h"
#include "robot_bridge/trajectory.h"

namespace robot_bridge {

struct PrimitiveOutput {
  MotionStatus status{MotionStatus::EXECUTING};
  Eigen::VectorXd q_cmd{};
  Eigen::VectorXd trq_cmd{};
  // This is only set with all cartesian mode primitives. MoveJ has this set to
  // I.
  Eigen::Isometry3d X_WT_cmd{Eigen::Isometry3d::Identity()};

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class MotionPrimitive {
public:
  enum Type {
    UNKNOWN = -1,
    MOVE_J = 0,
    MOVE_TOOL,
    MOVE_TOOL_STRAIGHT_UNTIL_TOUCH,
    HOLD_J_AND_APPLY_FORCE
  };

  MotionPrimitive(const std::string &name, const RigidBodyTree<double> *robot,
                  Type type);

  virtual ~MotionPrimitive() {
    std::cout << "[" << get_name() << "] exiting.\n";
  }

  const std::string &get_name() const { return name_; }
  double get_start_time() const { return start_time_; }
  double get_in_state_time(const RobotState &state) const {
    return state.get_time() - start_time_;
  }
  bool is_init() const { return init_; }

  void Initialize(const RobotState &state) {
    start_time_ = state.get_time();
    init_ = true;

    DoInitialize(state);

    std::cout << "[" << get_name() << "] initialized at " << state.get_time()
              << "\n";
  }

  void Control(const RobotState &state, PrimitiveOutput *output) const {
    // Set default values.
    const int dim = robot_.get_num_positions();
    output->q_cmd.resize(dim);
    output->trq_cmd.resize(dim);
    output->q_cmd.setZero();
    output->trq_cmd.setZero();

    output->X_WT_cmd.setIdentity();

    DoControl(state, output);
    output->status = ComputeStatus(state);
  }

  const RigidBodyTree<double> &get_robot() const { return robot_; }

  void set_name(const std::string &name) { name_ = name; }

  virtual void Update(const RobotState &) {}
  virtual void UpdateToolGoal(const RobotState &, const Eigen::Isometry3d &) {}
  virtual Eigen::Isometry3d GetToolGoal() const {
    return Eigen::Isometry3d::Identity();
  }

  const Eigen::VectorXd &get_velocity_upper_limit() const { return v_upper_; }
  const Eigen::VectorXd &get_velocity_lower_limit() const { return v_lower_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  virtual MotionStatus ComputeStatus(const RobotState &state) const = 0;
  virtual void DoInitialize(const RobotState &) {}
  virtual void DoControl(const RobotState &, PrimitiveOutput *) const {}
  Type get_type() const { return type_; }

private:
  const RigidBodyTree<double> &robot_;
  Eigen::VectorXd v_upper_{};
  Eigen::VectorXd v_lower_{};

  std::string name_{};
  const Type type_{UNKNOWN};
  double start_time_{0};
  bool init_{false};
};

class MoveJoint : public MotionPrimitive {
public:
  // Traj is exactly duration long.
  MoveJoint(const std::string &name, const RigidBodyTree<double> *robot,
            const Eigen::VectorXd &q0, const Eigen::VectorXd &q1,
            double duration);

  MoveJoint(const std::string &name, const RigidBodyTree<double> *robot,
            const Eigen::VectorXd &q0,
            const std::vector<Eigen::VectorXd> &q_des,
            const std::vector<double> &duration);

  // Traj duration will be determined by a line search, which would
  // satisfy the velocity and acceleration constraints.
  MoveJoint(const std::string &name, const RigidBodyTree<double> *robot,
            const Eigen::VectorXd &q0, const Eigen::VectorXd &q1);

  void DoControl(const RobotState &state, PrimitiveOutput *output)
    const override;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MotionStatus ComputeStatus(const RobotState &state) const override;

  PiecewisePolynomial<double> traj_;
  PiecewisePolynomial<double> trajd_;
};

class MoveTool : public MotionPrimitive {
public:
  void Update(const RobotState &state) override;

  virtual Eigen::Isometry3d
  ComputeDesiredToolInWorld(const RobotState &state) const = 0;

  Eigen::Isometry3d get_X_WT_ik() const {
    return get_robot().CalcFramePoseInWorldFrame(cache_, frame_T_);
  }

  void set_nominal_q(const Eigen::VectorXd &q_norm) { q_norm_ = q_norm; }
  void set_tool_gain(const Eigen::Vector6d &gain) { gain_T_ = gain; }
  const Eigen::Vector6d &get_tool_gain() const { return gain_T_; }

  const RigidBodyFrame<double> &get_tool_frame() const { return frame_T_; }

  void AddCollisionPair(const Capsule& c0, const Capsule& c1);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  MoveTool(const std::string &name, const RigidBodyTree<double> *robot,
           const RigidBodyFrame<double> *frame_T, const Eigen::VectorXd &q0,
           MotionPrimitive::Type type,
           const Eigen::Vector6d& F_thresh);

  void DoInitialize(const RobotState &state) override;
  void DoControl(const RobotState &state, PrimitiveOutput *output) const override;

  const JacobianIk &get_planner() const {
    return jaco_planner_;
  }
  const KinematicsCache<double> &get_cache() const { return cache_; }
  const RigidBodyFrame<double> &get_frame_T() const { return frame_T_; }
  bool is_stuck() const { return is_stuck_; }

  const Eigen::Vector6d& get_F_thresh() const { return F_thresh_; }
  bool is_F_over_thresh(const Eigen::Vector6d& F) const;

private:
  const Eigen::Vector6d F_thresh_;
  const RigidBodyFrame<double> frame_T_;
  KinematicsCache<double> cache_;
  bool is_stuck_{false};
  JacobianIk jaco_planner_;
  std::vector<std::pair<Capsule, Capsule>> collisions_;

  Eigen::Vector6d X_WT_diff_{Eigen::Vector6d::Zero()};
  Eigen::Vector6d gain_T_{Eigen::Vector6d::Constant(1)};
  Eigen::VectorXd q_norm_;
  double last_time_;
};

class MoveToolStraightUntilTouch : public MoveTool {
public:
  MoveToolStraightUntilTouch(const std::string &name,
                             const RigidBodyTree<double> *robot,
                             const RigidBodyFrame<double> *frame_T,
                             const Eigen::VectorXd &q0,
                             const Eigen::Vector3d &dir, double vel,
                             const Eigen::Vector6d& F_thresh);

  Eigen::Isometry3d
  ComputeDesiredToolInWorld(const RobotState &state) const override;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MotionStatus ComputeStatus(const RobotState &state) const override;

  Eigen::Isometry3d X_WT0_;
  Eigen::Vector3d dir_;
  double vel_;
};

class HoldPositionAndApplyForce : public MotionPrimitive {
public:
  HoldPositionAndApplyForce(const std::string &name,
                            const RigidBodyTree<double> *robot,
                            const RigidBodyFrame<double> *frame_T);

  void Update(const RobotState &state) override;

  const Eigen::Vector6d &get_desired_ext_wrench() const {
    return ext_wrench_d_;
  }
  void set_desired_ext_wrench(const Eigen::Vector6d &w) { ext_wrench_d_ = w; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  void DoInitialize(const RobotState &state) override;
  void DoControl(const RobotState &state, PrimitiveOutput *output) const override;
  MotionStatus ComputeStatus(const RobotState &state) const override;

  Eigen::VectorXd q0_;
  Eigen::Isometry3d X_WT0_;
  Eigen::Vector6d ext_wrench_d_{Eigen::Vector6d::Zero()};
  const RigidBodyFrame<double> frame_T_;
  KinematicsCache<double> cache_;
};

class MoveToolFollowTraj : public MoveTool {
public:
  MoveToolFollowTraj(
      const std::string &name, const RigidBodyTree<double> *robot,
      const RigidBodyFrame<double> *frame_T, const Eigen::VectorXd &q0,
      const drake::manipulation::SingleSegmentCartesianTrajectory<double> &traj,
      //const drake::manipulation::PiecewiseCartesianTrajectory<double> &traj,
      const Eigen::Vector6d& F_thresh);

  void set_X_WT_traj(
      const drake::manipulation::SingleSegmentCartesianTrajectory<double> &traj) {
      //const drake::manipulation::PiecewiseCartesianTrajectory<double> &traj) {
    X_WT_traj_ = traj;
  }

  void Update(const RobotState &state) override;

  Eigen::Isometry3d
  ComputeDesiredToolInWorld(const RobotState &state) const override;

  void set_applied_F(const Eigen::Vector6d& F) { F_ = F; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  void UpdateToolGoal(const RobotState &state,
                      const Eigen::Isometry3d &world_update) override;

  Eigen::Isometry3d GetToolGoal() const override {
    const double end_time = X_WT_traj_.get_position_trajectory().get_end_time();
    return X_WT_traj_.get_pose(end_time);
  }

  void DoControl(const RobotState &state, PrimitiveOutput *output) const override;
  MotionStatus ComputeStatus(const RobotState &state) const override;

  drake::manipulation::SingleSegmentCartesianTrajectory<double> X_WT_traj_;
  //drake::manipulation::PiecewiseCartesianTrajectory<double> X_WT_traj_;

  Eigen::Vector6d F_{Eigen::Vector6d::Zero()};
};

} // namespace robot_bridge
