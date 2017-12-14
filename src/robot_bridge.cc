#include "robot_bridge/robot_bridge.h"

namespace robot_bridge {

RobotBridge::RobotBridge(const RigidBodyTree<double> &tree,
                         const RigidBodyFrame<double> &tool_frame,
                         const RigidBodyFrame<double> &camera_frame)
    : robot_(tree), frame_T_(tool_frame), frame_C_(camera_frame) {}

bool RobotBridge::CheckJointLimitRadians(const Eigen::VectorXd &q_rad) const {
  if (q_rad.size() != robot_.get_num_positions()) {
    return false;
  }

  for (int i = 0; i < q_rad.size(); i++) {
    if (q_rad[i] < robot_.joint_limit_min[i] ||
        q_rad[i] > robot_.joint_limit_max[i]) {
      std::cout << "Joint[" << i << "] is out of limit: " << q_rad[i] << "["
                << robot_.joint_limit_min[i] << ", "
                << robot_.joint_limit_max[i] << "]" << std::endl;
      return false;
    }
  }
  return true;
}

MotionStatus RobotBridge::MoveJointDegrees(const Eigen::VectorXd &q,
                                           double duration, bool blocking) {
  return MoveJointRadians(q * M_PI / 180., duration, blocking);
}

MotionStatus RobotBridge::MoveJointDegrees(const Eigen::VectorXd &q,
                                           bool blocking) {
  return MoveJointRadians(q * M_PI / 180., blocking);
}

MotionStatus
RobotBridge::MoveJointDegrees(const std::vector<Eigen::VectorXd> &qs,
                              const std::vector<double> &durations,
                              bool blocking) {
  std::vector<Eigen::VectorXd> q_radians = qs;
  for (auto &q_radian : q_radians) {
    q_radian *= M_PI / 180.;
  }
  return MoveJointRadians(q_radians, durations, blocking);
}

// "GetC". Get cartesian pose and print on terminal.
Eigen::Isometry3d RobotBridge::GetToolPose() const {
  RobotState robot_state(&robot_, &frame_T_);
  GetRobotState(&robot_state);
  return robot_state.get_X_WP(frame_T_);
}

Eigen::Isometry3d RobotBridge::GetCameraPose() const {
  RobotState robot_state(&robot_, &frame_T_);
  GetRobotState(&robot_state);
  return robot_state.get_X_WP(frame_C_);
}

Eigen::VectorXd RobotBridge::GetJointPositionRadians() const {
  RobotState robot_state(&robot_, &frame_T_);
  GetRobotState(&robot_state);
  return robot_state.get_q();
}

Eigen::VectorXd RobotBridge::GetJointVelocityRadians() const {
  RobotState robot_state(&robot_, &frame_T_);
  GetRobotState(&robot_state);
  return robot_state.get_v();
}

Eigen::VectorXd RobotBridge::GetJointPositionDegrees() const {
  return GetJointPositionRadians() * 180. / M_PI;
}

Eigen::VectorXd RobotBridge::GetJointVelocityDegrees() const {
  return GetJointVelocityRadians() * 180. / M_PI;
}

MotionStatus RobotBridge::WaitForRobotMotionCompletion() const {
  MotionStatus ret;
  do {
    ret = GetRobotMotionStatus();
  } while (ret == MotionStatus::EXECUTING);

  switch (ret) {
    case MotionStatus::ERR_FORCE_SAFETY:
      std::cout << "Motion Ended with ERR_FORCE_SAFETY\n";
      break;
    case MotionStatus::ERR_STUCK:
      std::cout << "Motion Ended with ERR_STUCK\n";
      break;
    case MotionStatus::DONE:
      std::cout << "Motion Ended with DONE\n";
      break;
  }
  return ret;
}

} // namespace robot_bridge
