#pragma once

#include "robot_bridge/motion_primitive.h"
#include "robot_bridge/robot_bridge.h"

#include <lcm/lcm-cpp.hpp>

namespace robot_bridge {

class IiwaController final : public RobotBridge {
public:
  static const std::string kLcmIiwaStatusChannel;
  static const std::string kLcmIiwaCommandChannel;
  static const std::string kLcmIiwaControllerDebug;
  static const std::string kLcmWsgStatusChannel;
  static const std::string kLcmWsgCommandChannel;

  IiwaController(const RigidBodyTree<double> &robot,
                 const RigidBodyFrame<double> &tool_frame,
                 const RigidBodyFrame<double> &camera_frame);

  // Starts the controller thread, needs to be called before GetRobotState,
  // GetPrimitiveOutput, or exectuing any motions.
  void Start() override;
  void Stop() override;

  // Robot control.
  MotionStatus MoveJointRadians(const Eigen::VectorXd &q, double duration,
                                bool blocking) override;
  MotionStatus MoveJointRadians(const std::vector<Eigen::VectorXd> &qs,
                                const std::vector<double> &durations,
                                bool blocking) override;
  MotionStatus MoveJointRadians(const Eigen::VectorXd &q,
                                bool blocking) override;

  MotionStatus MoveToolAndApplyWrench(const Eigen::Isometry3d &tgt_pose_ee,
                                      const Eigen::Vector6d &gains_E,
                                      double duration,
                                      const Eigen::Vector6d &F_upper,
                                      const Eigen::Vector6d &F_lower,
                                      const Eigen::Vector6d &F,
                                      bool blocking) override;
  MotionStatus MoveStraightUntilTouch(const Eigen::Vector3d &dir_W, double vel,
                                      const Eigen::Vector3d &f_upper,
                                      const Eigen::Vector3d &f_lower,
                                      bool blocking) override;
  MotionStatus GetRobotMotionStatus() const override;
  Eigen::Isometry3d GetDesiredToolPose() const;
  Eigen::VectorXd GetDesiredJointPositionRadians() const;

  // Gripper control.
  bool CloseGripper() override;
  bool OpenGripper() override;
  bool CheckGrasp() const override;

  // State querry.
  // This blocks until there is at least 1 valid status message from iiwa.
  void GetRobotState(RobotState *state) const override;
  // This blocks until there is at least 1 valid status message from iiwa.
  void GetGripperState(GripperState *state) const override;

  // Controller querry.
  void GetPrimitiveOutput(PrimitiveOutput *output) const;

  void UpdateToolGoal(const Eigen::Isometry3d &world_frame_update) override;

  void AddCollisionPair(const Capsule &c0, const Capsule &c1) override;

private:
  void MoveJ(const Eigen::VectorXd &q_des, double duration);
  void MoveJ(const std::vector<Eigen::VectorXd> &q_des,
             const std::vector<double> &duration);

  void MoveJ(const Eigen::VectorXd &q_des);

  // position is in [m], force is in N. should be positive absolute value.
  void SetGripperPositionAndForce(double position, double force);

  inline void SwapPlan(std::unique_ptr<MotionPrimitive> new_plan) {
    std::lock_guard<std::mutex> guard(motion_lock_);
    primitive_ = std::move(new_plan);
    primitive_output_.status = MotionStatus::EXECUTING;
  }

  void ControlLoop();

  void HandleIiwaStatus(const lcm::ReceiveBuffer *, const std::string &,
                        const drake::lcmt_iiwa_status *status);
  void HandleWsgStatus(const lcm::ReceiveBuffer *, const std::string &,
                       const drake::lcmt_schunk_wsg_status *status);

  lcm::LCM lcm_;

  // These should live inside RBT..
  Eigen::VectorXd v_upper_;
  Eigen::VectorXd v_lower_;

  mutable std::mutex state_lock_;
  drake::lcmt_iiwa_status iiwa_status_{};
  drake::lcmt_schunk_wsg_status wsg_status_{};
  int iiwa_msg_ctr_{0};
  int wsg_msg_ctr_{0};

  mutable std::mutex motion_lock_;
  std::unique_ptr<MotionPrimitive> primitive_;
  PrimitiveOutput primitive_output_;

  std::thread control_thread_;
  std::atomic<bool> run_flag_{false};
  std::atomic<bool> ready_flag_{false};

  std::vector<std::pair<Capsule, Capsule>> collisions_;
};

} // namespace robot_bridge
