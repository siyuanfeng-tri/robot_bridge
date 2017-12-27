#include "robot_bridge/motion_primitive.h"

#include "drake/common/drake_assert.h"
#include "drake/util/drakeUtil.h"

namespace robot_bridge {

MotionPrimitive::MotionPrimitive(const std::string &name,
                                 const RigidBodyTree<double> *robot,
                                 const Eigen::VectorXd& v_u,
                                 const Eigen::VectorXd& v_l)
    : robot_(*robot), v_upper_(v_u), v_lower_(v_l), name_(name) {
  if (v_upper_.size() != robot_.get_num_velocities() ||
      v_lower_.size() != robot_.get_num_velocities() ||
      (v_upper_.array() < v_lower_.array()).any()) {
    throw std::runtime_error("invalid velocity limits.");
  }
}

///////////////////////////////////////////////////////////
MoveJoint::MoveJoint(const std::string &name,
                     const RigidBodyTree<double> *robot,
                     const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
                     const Eigen::VectorXd &q0, const Eigen::VectorXd &q1,
                     double duration)
    : MotionPrimitive(name, robot, v_u, v_l) {
  DRAKE_DEMAND(q0.size() == q1.size());
  DRAKE_DEMAND(q0.size() == get_robot().get_num_positions());
  std::vector<double> times = {0, duration};
  std::vector<Eigen::MatrixXd> knots = {q0, q1};
  Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(q0.size(), 1);
  traj_ = PiecewisePolynomial<double>::Cubic(times, knots, zero, zero);
  trajd_ = traj_.derivative();
}

MoveJoint::MoveJoint(const std::string &name,
                     const RigidBodyTree<double> *robot,
                     const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
                     const Eigen::VectorXd &q0,
                     const std::vector<Eigen::VectorXd> &q_des,
                     const std::vector<double> &duration)
    : MotionPrimitive(name, robot, v_u, v_l) {
  DRAKE_DEMAND(q0.size() == get_robot().get_num_positions());
  DRAKE_DEMAND(duration.size() == q_des.size());
  const int N = static_cast<int>(duration.size()) + 1;
  std::vector<double> times(N, 0);
  std::vector<Eigen::MatrixXd> knots(N, q0);
  for (int i = 1; i < N; i++) {
    times[i] = times[i - 1] + duration[i - 1];
    knots[i] = q_des[i - 1];
  }
  Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(q0.size(), 1);
  traj_ = PiecewisePolynomial<double>::Cubic(times, knots, zero, zero);
  trajd_ = traj_.derivative();
}

MoveJoint::MoveJoint(const std::string &name,
                     const RigidBodyTree<double> *robot,
                     const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
                     const Eigen::VectorXd &q0, const Eigen::VectorXd &q1)
    : MotionPrimitive(name, robot, v_u, v_l) {
  DRAKE_DEMAND(q0.size() == q1.size());
  DRAKE_DEMAND(q0.size() == get_robot().get_num_positions());
  std::vector<Eigen::MatrixXd> knots = {q0, q1};
  traj_ = RetimeTrajCubic(
      knots, Eigen::VectorXd::Zero(q0.size()), Eigen::VectorXd::Zero(q0.size()),
      get_velocity_lower_limit(), get_velocity_upper_limit(),
      Eigen::VectorXd::Constant(q0.size(), -8),
      Eigen::VectorXd::Constant(q0.size(), 8));
  trajd_ = traj_.derivative();
}

void MoveJoint::DoControl(const RobotState &state, PrimitiveOutput *output) const {
  const double interp_time = get_in_state_time(state);
  output->q_cmd = traj_.value(interp_time);
}

MotionStatus MoveJoint::ComputeStatus(const RobotState &state) const {
  if (get_in_state_time(state) > (traj_.getEndTime())) {
    return MotionStatus::DONE;
  } else {
    return MotionStatus::EXECUTING;
  }
}

///////////////////////////////////////////////////////////
MoveTool::MoveTool(const std::string &name, const RigidBodyTree<double> *robot,
                   const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
                   const RigidBodyFrame<double> *frame_T,
                   const Eigen::VectorXd &q0,
                   const Eigen::Vector6d& F_upper,
                   const Eigen::Vector6d& F_lower)
    : MotionPrimitive(name, robot, v_u, v_l), F_upper_(F_upper), F_lower_(F_lower),
      frame_T_(*frame_T), cache_(robot->CreateKinematicsCache()),
      jaco_planner_(robot), q_norm_(robot->getZeroConfiguration()) {
  cache_.initialize(q0, Eigen::VectorXd::Zero(robot->get_num_velocities()));
  get_robot().doKinematics(cache_);

  jaco_planner_.SetJointSpeedLimit(get_velocity_upper_limit(),
                                   get_velocity_lower_limit());

  if ((F_upper_.array() < F_lower_.array()).any()) {
    throw std::runtime_error("F upper is smaller than F lower.");
  }
}

bool MoveTool::is_F_over_thresh(const Eigen::Vector6d& F) const {
  return (F.array() > F_upper_.array()).any() |
         (F.array() < F_lower_.array()).any();
}

void MoveTool::AddCollisionPair(const Capsule& c0, const Capsule& c1) {
  collisions_.push_back(std::pair<Capsule, Capsule>(c0, c1));
  auto& pair = collisions_.back();
  const RigidBodyTree<double>& robot = get_robot();
  pair.first.set_pose(robot.CalcFramePoseInWorldFrame(cache_, pair.first.get_frame()));
  pair.second.set_pose(robot.CalcFramePoseInWorldFrame(cache_, pair.second.get_frame()));
}

void MoveTool::DoInitialize(const RobotState &state) {
  last_time_ = state.get_time();
}

void MoveTool::Update(const RobotState &state) {
  const RigidBodyTree<double>& robot = get_robot();
  Eigen::Isometry3d X_WT_d = ComputeDesiredToolInWorld(state);
  Eigen::Isometry3d X_WT =
      robot.CalcFramePoseInWorldFrame(cache_, frame_T_);

  const double dt = state.get_time() - last_time_;
  last_time_ = state.get_time();

  if (dt <= 0)
    return;

  Eigen::Vector6d V_WT_d =
      jaco_planner_.ComputePoseDiffInWorldFrame(X_WT, X_WT_d) / dt;

  Eigen::VectorXd v = jaco_planner_.ComputeDofVelocity(
      cache_, collisions_,
      frame_T_, V_WT_d,
      dt,
      q_norm_, cache_.getV(),
      &is_stuck_,
      gain_T_);

  // Integrate ik's fake state.
  cache_.initialize(cache_.getQ() + v * dt, v);
  robot.doKinematics(cache_);

  for (auto& pair : collisions_) {
    pair.first.set_pose(robot.CalcFramePoseInWorldFrame(cache_, pair.first.get_frame()));
    pair.second.set_pose(robot.CalcFramePoseInWorldFrame(cache_, pair.second.get_frame()));
  }
}

void MoveTool::DoControl(const RobotState &, PrimitiveOutput *output) const {
  output->q_cmd = cache_.getQ();
  output->X_WT_cmd = get_robot().CalcFramePoseInWorldFrame(cache_, frame_T_);
}

///////////////////////////////////////////////////////////
MoveToolStraightUntilTouch::MoveToolStraightUntilTouch(
    const std::string &name, const RigidBodyTree<double> *robot,
    const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
    const RigidBodyFrame<double> *frame_T, const Eigen::VectorXd &q0,
    const Eigen::Vector3d &dir, double vel,
    const Eigen::Vector6d& F_upper,
    const Eigen::Vector6d& F_lower)
    : MoveTool(name, robot, v_u, v_l, frame_T, q0, F_upper, F_lower),
      X_WT0_{get_X_WT_ik()}, dir_{dir.normalized()}, vel_{vel} {}

void MoveToolStraightUntilTouch::Update(const RobotState &state) {
  if (is_F_over_thresh(state.get_ext_wrench()) && !stopped_) {
    stopped_ = true;
    X_WT_stopped_ = get_X_WT_ik();
  }

  MoveTool::Update(state);
}

Eigen::Isometry3d MoveToolStraightUntilTouch::ComputeDesiredToolInWorld(
    const RobotState &state) const {
  Eigen::Isometry3d ret = X_WT0_;
  if (!stopped_) {
    ret.translation() += dir_ * vel_ * get_in_state_time(state);
  }
  else {
    ret.translation() = X_WT_stopped_.translation();
  }

  return ret;
}

MotionStatus
MoveToolStraightUntilTouch::ComputeStatus(const RobotState &state) const {
  if (is_F_over_thresh(state.get_ext_wrench()))
    return MotionStatus::DONE;
  else
    return MotionStatus::EXECUTING;
}

///////////////////////////////////////////////////////////
HoldPositionAndApplyForce::HoldPositionAndApplyForce(
    const std::string &name, const RigidBodyTree<double> *robot,
    const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
    const RigidBodyFrame<double> *frame_T)
    : MotionPrimitive(name, robot, v_u, v_l), frame_T_(*frame_T),
      cache_(robot->CreateKinematicsCache()) {}

void HoldPositionAndApplyForce::Update(const RobotState &state) {
  cache_.initialize(state.get_q(), state.get_v());
  get_robot().doKinematics(cache_);
}

void HoldPositionAndApplyForce::DoInitialize(const RobotState &state) {
  q0_ = state.get_q();
  KinematicsCache<double> tmp = get_robot().CreateKinematicsCache();
  tmp.initialize(q0_);
  get_robot().doKinematics(tmp);
  X_WT0_ = get_robot().CalcFramePoseInWorldFrame(tmp, frame_T_);
}

void HoldPositionAndApplyForce::DoControl(
    const RobotState &, PrimitiveOutput *output) const {
  const RigidBodyTree<double> &robot = get_robot();
  output->q_cmd = q0_;
  output->X_WT_cmd = X_WT0_;

  // ext_trq = trq_measured - trq_id
  //         = M * qdd + h - J^T * F - (M * qdd + h)
  //         = -J^T * F
  Eigen::MatrixXd J =
      robot.CalcFrameSpatialVelocityJacobianInWorldFrame(cache_, frame_T_);
  output->trq_cmd = -J.transpose() * ext_wrench_d_;
}

MotionStatus
HoldPositionAndApplyForce::ComputeStatus(const RobotState &state) const {
  return MotionStatus::EXECUTING;
}

///////////////////////////////////////////////////////////
MoveToolFollowTraj::MoveToolFollowTraj(
    const std::string &name, const RigidBodyTree<double> *robot,
    const Eigen::VectorXd& v_u, const Eigen::VectorXd& v_l,
    const RigidBodyFrame<double> *frame_T, const Eigen::VectorXd &q0,
    const drake::manipulation::SingleSegmentCartesianTrajectory<double> &traj,
    //const drake::manipulation::PiecewiseCartesianTrajectory<double> &traj,
    const Eigen::Vector6d& F_upper,
    const Eigen::Vector6d& F_lower)
    : MoveTool(name, robot, v_u, v_l, frame_T, q0, F_upper, F_lower),
      X_WT_traj_(traj) {}

Eigen::Isometry3d
MoveToolFollowTraj::ComputeDesiredToolInWorld(const RobotState &state) const {
  const double interp_t = get_in_state_time(state);
  return X_WT_traj_.get_pose(interp_t);
}

void MoveToolFollowTraj::Update(const RobotState &state) {
  if (is_F_over_thresh(state.get_ext_wrench()) && !stopped_) {
    const double end_time = get_in_state_time(state);
    const double start_time =
        X_WT_traj_.get_position_trajectory().get_start_time();
    Eigen::Isometry3d X_WT0 = X_WT_traj_.get_pose(start_time);
    Eigen::Isometry3d X_WT1 = get_X_WT_ik();

    /*
    X_WT_traj_ = drake::manipulation::PiecewiseCartesianTrajectory<
        double>::MakeCubicLinearWithEndLinearVelocity({start_time, end_time},
                                                      {X_WT0, X_WT1},
                                                      Eigen::Vector3d::Zero(),
                                                      Eigen::Vector3d::Zero());
    */
    X_WT_traj_ = drake::manipulation::SingleSegmentCartesianTrajectory<double>(
        X_WT0, X_WT1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        0, 0,
        start_time, end_time);
    stopped_ = true;
  }

  MoveTool::Update(state);
}

void MoveToolFollowTraj::UpdateToolGoal(const RobotState &state,
                                        const Eigen::Isometry3d &new_goal) {
  const double interp_t = get_in_state_time(state);
  Eigen::Vector6d V_WT = X_WT_traj_.get_velocity(interp_t);
  Eigen::Isometry3d X_WT = X_WT_traj_.get_pose(interp_t);

  double end_time = X_WT_traj_.get_position_trajectory().get_end_time();
  double start_time = interp_t;

  Eigen::Vector6d diff =
      JacobianIk::ComputePoseDiffInWorldFrame(X_WT, new_goal);

  double dt_lin = diff.tail<3>().norm() / 0.1;
  double dt_ang = diff.head<3>().norm() / 0.5;
  double min_dt = std::max(dt_lin, dt_ang);
  min_dt = std::max(min_dt, 0.2);

  end_time = std::max(end_time, start_time + min_dt);

  /*
  auto traj = drake::manipulation::PiecewiseCartesianTrajectory<
      double>::MakeCubicLinearWithEndLinearVelocity({start_time, end_time},
                                                    {X_WT, new_goal},
                                                    V_WT.tail<3>(),
                                                    Eigen::Vector3d::Zero());
  */

  // this is sketchy
  auto traj = drake::manipulation::SingleSegmentCartesianTrajectory<double>(
      X_WT, new_goal, V_WT.tail<3>(), Eigen::Vector3d::Zero(),
      0, 0,
      start_time, end_time);
  X_WT_traj_ = traj;
}

void MoveToolFollowTraj::DoControl(const RobotState &state,
                                   PrimitiveOutput *output) const {
  // Gets the current actual jacobian.
  Eigen::MatrixXd J = get_robot().CalcFrameSpatialVelocityJacobianInWorldFrame(
      state.get_cache(), get_tool_frame());

  // The desired q comes from MoveTool's
  MoveTool::DoControl(state, output);

  // Apply static external force.
  output->trq_cmd = -J.transpose() * F_;
}

MotionStatus MoveToolFollowTraj::ComputeStatus(const RobotState &state) const {
  const double duration = X_WT_traj_.get_position_trajectory().get_end_time();

  Eigen::Isometry3d X_WT_d = ComputeDesiredToolInWorld(state);
  Eigen::Isometry3d X_WT = get_X_WT_ik();

  Eigen::Vector6d V_WT_d =
      JacobianIk::ComputePoseDiffInWorldFrame(X_WT, X_WT_d);
  Eigen::Vector6d diff;
  diff.head<3>() = X_WT.linear().transpose() * V_WT_d.head<3>();
  diff.tail<3>() = X_WT.linear().transpose() * V_WT_d.tail<3>();
  diff = (diff.array() * get_tool_gain().array()).matrix();

  // const Eigen::VectorXd& ik_v = get_cache().getV();

  if (is_F_over_thresh(state.get_ext_wrench())) {
    return MotionStatus::ERR_FORCE_SAFETY;
  }

  if (diff.norm() < 1e-3 && get_in_state_time(state) > duration) {
    return MotionStatus::DONE;
  }

  // if (is_stuck() || (state.get_v().norm() < 1e-2 && get_in_state_time(state) > 3 * duration)) {
  if (state.get_v().norm() < 1e-3 &&
      get_in_state_time(state) > 2 * duration) {
    return MotionStatus::ERR_STUCK;
  }
  if (is_stuck()) {
    return MotionStatus::ERR_STUCK;
  }

  return MotionStatus::EXECUTING;
}

} // namespace robot_bridge
