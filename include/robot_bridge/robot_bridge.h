#pragma once

#include "drake/multibody/rigid_body_tree.h"
#include "robot_bridge/capsule.h"
#include "robot_bridge/robot_bridge_common.h"

namespace robot_bridge {

class RobotBridge {
public:
  RobotBridge(const RigidBodyTree<double> &tree,
              const RigidBodyFrame<double> &tool_frame,
              const RigidBodyFrame<double> &camera_frame);

  const RigidBodyTree<double> &get_robot() const { return robot_; }
  const RigidBodyFrame<double> &get_tool_frame() const { return frame_T_; }
  const RigidBodyFrame<double> &get_camera_frame() const { return frame_C_; }

  virtual void Start() = 0;
  virtual void Stop() = 0;

  // "SetTool". Set tool transform with respect to link 7. This could be the
  // gripper frame or the camera frame.
  void SetToolTransform(const Eigen::Isometry3d &tf){};

  // "MoveQ". Linear movement in joint space.
  virtual MotionStatus MoveJointRadians(const Eigen::VectorXd &q,
                                        double duration, bool blocking) = 0;
  MotionStatus MoveJointDegrees(const Eigen::VectorXd &q, double duration,
                                bool blocking);
  virtual MotionStatus MoveJointRadians(const std::vector<Eigen::VectorXd> &qs,
                                        const std::vector<double> &durations,
                                        bool blocking) = 0;
  MotionStatus MoveJointDegrees(const std::vector<Eigen::VectorXd> &qs,
                                const std::vector<double> &durations,
                                bool blocking);
  // Controller handling retiming.
  virtual MotionStatus MoveJointRadians(const Eigen::VectorXd &q,
                                        bool blocking) = 0;
  MotionStatus MoveJointDegrees(const Eigen::VectorXd &q, bool blocking);

  // "MoveL". Linear movement in cartesian space.
  virtual MotionStatus MoveTool(const Eigen::Isometry3d &tgt_pose_ee,
                                double duration, double Fz_thresh,
                                bool blocking) = 0;
  virtual MotionStatus MoveToolAndApplyFz(const Eigen::Isometry3d &tgt_pose_ee,
                                          double duration, double Fz_thresh,
                                          double Fz, double mu,
                                          bool blocking) = 0;
  virtual void UpdateToolGoal(const Eigen::Isometry3d &world_update) = 0;

  // Controller status querry.
  virtual Eigen::Isometry3d GetDesiredToolPose() const = 0;
  virtual Eigen::VectorXd GetDesiredJointPositionRadians() const = 0;
  // Checks if a motion primitive has finished.
  // Returns 0 < 0 if erred, > 0 if done.
  virtual MotionStatus GetRobotMotionStatus() const = 0;
  // Blocks until a motion primitive is done.
  MotionStatus WaitForRobotMotionCompletion() const;

  // Robot + gripper status querry.
  virtual void GetRobotState(RobotState *state) const = 0;
  virtual void GetGripperState(GripperState *state) const = 0;
  Eigen::VectorXd GetJointPositionRadians() const;
  Eigen::VectorXd GetJointVelocityRadians() const;
  Eigen::VectorXd GetJointPositionDegrees() const;
  Eigen::VectorXd GetJointVelocityDegrees() const;
  Eigen::Isometry3d GetToolPose() const;
  Eigen::Isometry3d GetCameraPose() const;

  // Gripper control. These two always blocks.
  virtual bool CloseGripper() = 0;
  virtual bool OpenGripper() = 0;
  // Gripper status querry.
  virtual bool CheckGrasp() const = 0;

  // Returns false if q_rad is out of limit defined by robot_.
  bool CheckJointLimitRadians(const Eigen::VectorXd &q_rad) const;

  virtual void AddCollisionPair(const Capsule& c0, const Capsule& c1) = 0;

private:
  const RigidBodyTree<double> &robot_;
  // Tool frame.
  const RigidBodyFrame<double> frame_T_;
  // Camera frame.
  const RigidBodyFrame<double> frame_C_;
};

} // namespace robot_bridge
