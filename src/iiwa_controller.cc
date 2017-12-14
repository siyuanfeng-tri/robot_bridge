#include "robot_bridge/iiwa_controller.h"

#include <fstream>
#include <iostream>
#include <memory>

#include <lcm/lcm-cpp.hpp>

#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_schunk_wsg_command.hpp"
#include "drake/lcmt_viewer_draw.hpp"

#include "drake/util/drakeUtil.h"
#include "drake/util/lcmUtil.h"
#include "robot_bridge/util.h"

namespace robot_bridge {

const std::string IiwaController::kLcmIiwaStatusChannel = "IIWA_STATUS";
const std::string IiwaController::kLcmIiwaCommandChannel = "IIWA_COMMAND";
const std::string IiwaController::kLcmIiwaControllerDebug = "CTRL_DEBUG";
const std::string IiwaController::kLcmWsgStatusChannel = "SCHUNK_WSG_STATUS";
const std::string IiwaController::kLcmWsgCommandChannel = "SCHUNK_WSG_COMMAND";

IiwaController::IiwaController(const RigidBodyTree<double> &robot,
                               const RigidBodyFrame<double> &tool_frame,
                               const RigidBodyFrame<double> &camera_frame)
    : RobotBridge(robot, tool_frame, camera_frame) {
  // Iiwa status.
  lcm::Subscription *sub = lcm_.subscribe(
      kLcmIiwaStatusChannel, &IiwaController::HandleIiwaStatus, this);
  // THIS IS VERY IMPORTANT!!
  sub->setQueueCapacity(1);

  // Gripper status.
  sub = lcm_.subscribe(kLcmWsgStatusChannel, &IiwaController::HandleWsgStatus,
                       this);
  sub->setQueueCapacity(1);
}

void IiwaController::Start() {
  if (run_flag_) {
    std::cout << "Controller thread already running\n";
    return;
  }
  run_flag_ = true;
  ready_flag_ = false;
  iiwa_msg_ctr_ = 0;
  wsg_msg_ctr_ = 0;

  control_thread_ = std::thread(&IiwaController::ControlLoop, this);

  // Block until ready.
  while (!ready_flag_)
    ;
}

void IiwaController::Stop() {
  if (!run_flag_) {
    std::cout << "Controller thread not running\n";
    return;
  }

  run_flag_ = false;
  ready_flag_ = false;
  control_thread_.join();
}

void IiwaController::AddCollisionPair(const Capsule& c0, const Capsule& c1) {
  collisions_.push_back(std::pair<Capsule, Capsule>(c0, c1));
}

MotionStatus IiwaController::MoveJointRadians(const Eigen::VectorXd &q,
                                              double duration, bool blocking) {
  if (!CheckJointLimitRadians(q))
    return MotionStatus::ERR_INVALID_ARG;

  MoveJ(q, duration);
  if (blocking)
    return WaitForRobotMotionCompletion();
  return GetRobotMotionStatus();
}

MotionStatus
IiwaController::MoveJointRadians(const std::vector<Eigen::VectorXd> &qs,
                                 const std::vector<double> &durations,
                                 bool blocking) {
  for (const auto &q : qs) {
    if (!CheckJointLimitRadians(q))
      return MotionStatus::ERR_INVALID_ARG;
  }

  MoveJ(qs, durations);
  if (blocking)
    return WaitForRobotMotionCompletion();
  return GetRobotMotionStatus();
}

MotionStatus IiwaController::MoveJointRadians(const Eigen::VectorXd &q,
                                              bool blocking) {
  if (!CheckJointLimitRadians(q))
    return MotionStatus::ERR_INVALID_ARG;

  MoveJ(q);
  if (blocking)
    return WaitForRobotMotionCompletion();
  return GetRobotMotionStatus();
}

MotionStatus
IiwaController::MoveToolAndApplyWrench(
    const Eigen::Isometry3d &tgt_pose_ee,
    double duration,
    const Eigen::Vector6d& F_thresh,
    const Eigen::Vector6d& F,
    bool blocking) {
  RobotState robot_state(&get_robot(), &get_tool_frame());
  GetRobotState(&robot_state);

  Eigen::Isometry3d cur_pose_ee = robot_state.get_X_WT();
  drake::manipulation::PiecewiseCartesianTrajectory<double> traj =
      drake::manipulation::PiecewiseCartesianTrajectory<double>::
          MakeCubicLinearWithEndLinearVelocity(
              {0.2, duration}, {cur_pose_ee, tgt_pose_ee},
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  MoveToolFollowTraj(traj, F_thresh, F);
  if (blocking)
    return WaitForRobotMotionCompletion();
  return GetRobotMotionStatus();
}

MotionStatus IiwaController::MoveStraightUntilTouch(
    const Eigen::Vector3d &dir_W,
    double vel, const Eigen::Vector3d& force_thresh,
    bool blocking) {
  PrimitiveOutput cur_output;
  GetPrimitiveOutput(&cur_output);

  Eigen::Vector6d thresh = Eigen::Vector6d::Constant(1000);
  thresh.tail<3>() = force_thresh;

  auto new_plan = new MoveToolStraightUntilTouch(
      "MoveStraightUntilTouch", &get_robot(), &get_tool_frame(),
      cur_output.q_cmd, dir_W, vel, thresh);
  SwapPlan(std::unique_ptr<MotionPrimitive>(new_plan));

  if (blocking)
    return WaitForRobotMotionCompletion();
  return GetRobotMotionStatus();
}

MotionStatus IiwaController::GetRobotMotionStatus() const {
  PrimitiveOutput output;
  GetPrimitiveOutput(&output);
  return output.status;
}

Eigen::Isometry3d IiwaController::GetDesiredToolPose() const {
  PrimitiveOutput output;
  GetPrimitiveOutput(&output);
  return output.X_WT_cmd;
}

Eigen::VectorXd IiwaController::GetDesiredJointPositionRadians() const {
  PrimitiveOutput output;
  GetPrimitiveOutput(&output);
  return output.q_cmd;
}

bool IiwaController::CloseGripper() {
  // mm
  double dist = 0.006;
  // N
  double force = 70;

  SetGripperPositionAndForce(dist, force);

  GripperState wsg_state;
  while (true) {
    GetGripperState(&wsg_state);
    // When grasped, force is negative.
    if (std::fabs(wsg_state.get_force() + force) < 1 &&
        std::fabs(wsg_state.get_velocity()) == 0) {
      return true;
    }
    // Or empty grasp,
    if (std::fabs(wsg_state.get_position() - dist) < 1e-4 &&
        std::fabs(wsg_state.get_velocity()) == 0) {
      return false;
    }
  }
}

bool IiwaController::OpenGripper() {
  // mm
  double dist = 0.107;
  // N
  double force = 40;
  SetGripperPositionAndForce(dist, force);
  GripperState wsg_state;

  double t0 = get_system_time();
  while (true) {
    GetGripperState(&wsg_state);
    double dt = get_system_time() - t0;
    if ((std::fabs(wsg_state.get_velocity()) == 0 &&
         std::fabs(wsg_state.get_position() - dist) < 1e-3)) {
      return true;
    }
    if (dt > 2.) {
      return false;
    }
  }
  return false;
}

bool IiwaController::CheckGrasp() const {
  double force = 70; // N
  GripperState wsg_state;
  for (int i = 0; i < 10; i++) {
    GetGripperState(&wsg_state);
    // Wsg returns - force when closing.
    if (std::fabs(wsg_state.get_force() + force) < 10 &&
        std::fabs(wsg_state.get_velocity()) < 3 &&
        wsg_state.get_position() >= 0.009) {
      return true;
    }
    std::cout << "NOT GRASPING: "
              << "f: " << wsg_state.get_force()
              << ", v: " << wsg_state.get_velocity()
              << ", p: " << wsg_state.get_position() << "\n";
  }
  return false;
}

void IiwaController::GetPrimitiveOutput(PrimitiveOutput *output) const {
  DRAKE_DEMAND(run_flag_ && ready_flag_);

  std::lock_guard<std::mutex> guard(motion_lock_);
  *output = primitive_output_;
}

void IiwaController::HandleIiwaStatus(const lcm::ReceiveBuffer *,
                                      const std::string &,
                                      const drake::lcmt_iiwa_status *status) {
  std::lock_guard<std::mutex> guard(state_lock_);
  iiwa_status_ = *status;
  iiwa_msg_ctr_++;
}

void IiwaController::HandleWsgStatus(
    const lcm::ReceiveBuffer *, const std::string &,
    const drake::lcmt_schunk_wsg_status *status) {
  std::lock_guard<std::mutex> guard(state_lock_);
  wsg_status_ = *status;
  wsg_msg_ctr_++;
}

void IiwaController::MoveJ(const Eigen::VectorXd &q_des, double duration) {
  PrimitiveOutput cur_output;
  GetPrimitiveOutput(&cur_output);

  std::unique_ptr<MotionPrimitive> new_plan(
      new MoveJoint("MoveJ", &get_robot(), cur_output.q_cmd, q_des, duration));
  SwapPlan(std::move(new_plan));
}

void IiwaController::MoveJ(const std::vector<Eigen::VectorXd> &q_des,
                           const std::vector<double> &duration) {
  PrimitiveOutput cur_output;
  GetPrimitiveOutput(&cur_output);

  std::unique_ptr<MotionPrimitive> new_plan(
      new MoveJoint("MoveJ", &get_robot(), cur_output.q_cmd, q_des, duration));
  SwapPlan(std::move(new_plan));
}

void IiwaController::MoveJ(const Eigen::VectorXd &q_des) {
  PrimitiveOutput cur_output;
  GetPrimitiveOutput(&cur_output);

  std::unique_ptr<MotionPrimitive> new_plan(
      new MoveJoint("MoveJ", &get_robot(), cur_output.q_cmd, q_des));
  SwapPlan(std::move(new_plan));
}

void IiwaController::MoveToolFollowTraj(
    const drake::manipulation::PiecewiseCartesianTrajectory<double> &traj,
    const Eigen::Vector6d& F_thresh,
    const Eigen::Vector6d& F) {
  PrimitiveOutput cur_output;
  GetPrimitiveOutput(&cur_output);

  auto new_plan = new class MoveToolFollowTraj(
      "MoveToolFollowTraj", &get_robot(), &get_tool_frame(), cur_output.q_cmd,
      traj, F_thresh);
  new_plan->set_applied_F(F);
  for (const auto& collision_pair : collisions_)
    new_plan->AddCollisionPair(collision_pair.first, collision_pair.second);
  SwapPlan(std::unique_ptr<MotionPrimitive>(new_plan));
}

void IiwaController::UpdateToolGoal(
    const Eigen::Isometry3d &world_frame_update) {
  RobotState robot_state(&get_robot(), &get_tool_frame());
  GetRobotState(&robot_state);

  std::lock_guard<std::mutex> guard(motion_lock_);
  primitive_->UpdateToolGoal(robot_state, world_frame_update);
}

void IiwaController::GetRobotState(RobotState *state) const {
  DRAKE_DEMAND(run_flag_);

  drake::lcmt_iiwa_status stats;
  while (true) {
    std::lock_guard<std::mutex> guard(state_lock_);
    if (iiwa_msg_ctr_ == 0) {
      std::cout << "hasn't got a valid iiwa state yet.\n";
    } else {
      stats = iiwa_status_;
      break;
    }
  }
  state->UpdateState(stats);
}

void IiwaController::GetGripperState(GripperState *state) const {
  DRAKE_DEMAND(run_flag_);

  drake::lcmt_schunk_wsg_status stats;
  while (true) {
    std::lock_guard<std::mutex> guard(state_lock_);
    if (wsg_msg_ctr_ == 0) {
      std::cout << "hasn't got a valid wsg state yet.\n";
    } else {
      stats = wsg_status_;
      break;
    }
  }
  state->UpdateState(stats);
}

void IiwaController::SetGripperPositionAndForce(double position, double force) {
  drake::lcmt_schunk_wsg_command cmd{};
  // TODO set timestamp.
  cmd.target_position_mm = position * 1e3;
  cmd.force = force;
  lcm_.publish(kLcmWsgCommandChannel, &cmd);
}

void FillFrameMessage(const Eigen::Isometry3d &pose, int idx,
                      drake::lcmt_viewer_draw *msg) {
  for (int j = 0; j < 3; j++)
    msg->position[idx][j] = static_cast<float>(pose.translation()[j]);

  Eigen::Quaterniond quat(pose.linear());
  msg->quaternion[idx][0] = static_cast<float>(quat.w());
  msg->quaternion[idx][1] = static_cast<float>(quat.x());
  msg->quaternion[idx][2] = static_cast<float>(quat.y());
  msg->quaternion[idx][3] = static_cast<float>(quat.z());
}

void IiwaController::ControlLoop() {
  const RigidBodyTree<double> &robot = get_robot();
  const RigidBodyFrame<double> &frame_T = get_tool_frame();
  const RigidBodyFrame<double> &frame_C = get_camera_frame();

  // We can directly read iiwa_status_ because write only happens in
  // HandleIiwaStatus, which is only triggered by calling lcm_.handle,
  // which is only called from this func (thus, in the same thread).

  ////////////////////////////////////////////////////////
  // state related
  RobotState state(&robot, &frame_T);

  // VERY IMPORTANT HACK, To make sure that no stale messages are in the
  // queue.
  int lcm_err;
  while (iiwa_msg_ctr_ < 4) {
    lcm_err = lcm_.handleTimeout(10);
  }
  std::cout << "got first iiwa msg\n";

  ////////////////////////////////////////////////////////
  // cmd related
  drake::lcmt_iiwa_command iiwa_command{};
  iiwa_command.num_joints = robot.get_num_positions();
  iiwa_command.joint_position.resize(robot.get_num_positions(), 0.);
  iiwa_command.num_torques = robot.get_num_positions();
  iiwa_command.joint_torque.resize(robot.get_num_positions(), 0.);

  Eigen::VectorXd q_cmd(7);
  Eigen::VectorXd trq_cmd = Eigen::VectorXd::Zero(7);
  Eigen::Isometry3d X_WT_cmd = Eigen::Isometry3d::Identity();

  // Make initial plan.
  DRAKE_DEMAND(state.UpdateState(iiwa_status_));
  for (int i = 0; i < robot.get_num_positions(); i++) {
    q_cmd[i] = iiwa_status_.joint_position_measured[i];
  }

  // Make initial plan to go to q1.
  auto plan = std::unique_ptr<MotionPrimitive>(
      new MoveJoint("hold_q", &robot, state.get_q(), state.get_q(), 0.1));
  SwapPlan(std::move(plan));
  {
    std::lock_guard<std::mutex> guard(motion_lock_);
    primitive_->Initialize(state);
    primitive_->Update(state);
    primitive_->Control(state, &primitive_output_);
  }
  ready_flag_ = true;

  // Frame visualziation stuff.
  Eigen::Isometry3d GRASP = Eigen::Isometry3d::Identity();

  drake::lcmt_viewer_draw frame_msg{};
  frame_msg.link_name = {"Camera", "Tool_measured", "Tool_ik", "Grasp"};
  frame_msg.num_links = frame_msg.link_name.size();
  // The robot num is not relevant here.
  frame_msg.robot_num.resize(frame_msg.num_links, 0);
  std::vector<float> pos = {0, 0, 0};
  std::vector<float> quaternion = {1, 0, 0, 0};
  frame_msg.position.resize(frame_msg.num_links, pos);
  frame_msg.quaternion.resize(frame_msg.num_links, quaternion);

  ////////////////////////////////////////////////////////
  // Main loop.
  while (run_flag_) {
    // Call lcm handle until at least one status message is
    // processed.
    do {
      lcm_err = lcm_.handleTimeout(10);
      if (lcm_err == 0) {
        std::cout << "LCM recv timed out in control loop 10ms.\n";
      } else if (lcm_err < 0) {
        std::cout << "LCM recv error.\n";
      } else {
        // Update state.
        if (state.UpdateState(iiwa_status_))
          break;
      }
    } while (lcm_err <= 0);

    // Locked by motion_lock_
    {
      std::lock_guard<std::mutex> guard(motion_lock_);
      if (!primitive_->is_init()) {
        primitive_->Initialize(state);
      }
      primitive_->Update(state);
      primitive_->Control(state, &primitive_output_);

      q_cmd = primitive_output_.q_cmd;
      trq_cmd = primitive_output_.trq_cmd;
      X_WT_cmd = primitive_output_.X_WT_cmd;

      GRASP = primitive_->GetToolGoal();
    }

    // Generate frame visualization stuff.
    frame_msg.timestamp = static_cast<int64_t>(state.get_time() * 1e6);
    FillFrameMessage(state.get_X_WP(frame_C), 0, &frame_msg);
    FillFrameMessage(state.get_X_WT(), 1, &frame_msg);
    FillFrameMessage(X_WT_cmd, 2, &frame_msg);
    FillFrameMessage(GRASP, 3, &frame_msg);
    lcm_.publish("DRAKE_DRAW_FRAMES", &frame_msg);

    // send command
    iiwa_command.utime = static_cast<int64_t>(state.get_time() * 1e6);
    for (int i = 0; i < robot.get_num_positions(); i++) {
      iiwa_command.joint_position[i] = q_cmd[i];
      iiwa_command.joint_torque[i] = trq_cmd[i];
    }
    lcm_.publish(kLcmIiwaCommandChannel, &iiwa_command);
  }
}

} // namespace robot_bridge
