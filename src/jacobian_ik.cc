#include "robot_bridge/jacobian_ik.h"
#include "robot_bridge/util.h"

#include <memory>

#include "drake/common/text_logging.h"
#include "drake/multibody/ik_options.h"
#include "drake/multibody/joints/floating_base_types.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_constraint.h"
#include "drake/multibody/rigid_body_ik.h"

#include "drake/solvers/mathematical_program.h"

namespace robot_bridge {

Eigen::Vector6d
JacobianIk::ComputePoseDiffInWorldFrame(const Eigen::Isometry3d &pose0,
                                        const Eigen::Isometry3d &pose1) {
  Eigen::Vector6d diff = Eigen::Vector6d::Zero();

  // Linear.
  diff.tail<3>() = (pose1.translation() - pose0.translation());

  // Angular.
  Eigen::AngleAxisd rot_err(pose1.linear() * pose0.linear().transpose());
  diff.head<3>() = rot_err.axis() * rot_err.angle();

  return diff;
}

void JacobianIk::Setup() {
  DRAKE_DEMAND(robot_->get_num_positions() == robot_->get_num_velocities());

  q_lower_ = robot_->joint_limit_min;
  q_upper_ = robot_->joint_limit_max;
  v_lower_ = Eigen::VectorXd::Constant(robot_->get_num_velocities(), -2);
  v_upper_ = Eigen::VectorXd::Constant(robot_->get_num_velocities(), 2);
  unconstrained_dof_v_limit_ = Eigen::VectorXd::Constant(1, 0.6);

  identity_ = Eigen::MatrixXd::Identity(robot_->get_num_positions(),
                                        robot_->get_num_positions());
  zero_ = Eigen::VectorXd::Zero(robot_->get_num_velocities());
}

void JacobianIk::SetJointSpeedLimit(const Eigen::VectorXd &v_upper,
                                    const Eigen::VectorXd &v_lower) {
  DRAKE_DEMAND(v_upper.size() == v_lower.size());
  DRAKE_DEMAND(v_lower.size() == v_lower_.size());
  v_lower_ = v_lower;
  v_upper_ = v_upper;
}

JacobianIk::JacobianIk(const RigidBodyTree<double> *robot) : robot_{robot} {
  Setup();
}

JacobianIk::JacobianIk(const std::string &model_path,
                       const Eigen::Isometry3d &base_to_world) {
  auto base_frame = std::allocate_shared<RigidBodyFrame<double>>(
      Eigen::aligned_allocator<RigidBodyFrame<double>>(), "world", nullptr,
      base_to_world);

  owned_robot_ = std::make_unique<RigidBodyTree<double>>();
  robot_ = owned_robot_.get();

  drake::parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path, drake::multibody::joints::kFixed, base_frame, owned_robot_.get());

  Setup();
}

Eigen::VectorXd JacobianIk::ComputeDofVelocity(
    const KinematicsCache<double> &cache,
    const std::vector<std::pair<Capsule, Capsule>>& collisions,
    const RigidBodyFrame<double> &frame_E, const Eigen::Vector6d &V_WE,
    const Eigen::VectorXd &q_nominal, double dt, bool *is_stuck,
    const Eigen::Vector6d &gain_E) const {
  DRAKE_DEMAND(q_nominal.size() == robot_->get_num_positions());
  DRAKE_DEMAND(dt > 0);

  Eigen::VectorXd ret;

  drake::solvers::MathematicalProgram prog;
  drake::solvers::VectorXDecisionVariable v =
      prog.NewContinuousVariables(robot_->get_num_velocities(), "v");
  drake::solvers::VectorXDecisionVariable alpha =
      prog.NewContinuousVariables(1, "alpha");

  // Add ee vel constraint.
  Eigen::Isometry3d X_WE = robot_->CalcFramePoseInWorldFrame(cache, frame_E);

  drake::Matrix6<double> R_EW = drake::Matrix6<double>::Zero();
  R_EW.block<3, 3>(0, 0) = X_WE.linear().transpose();
  R_EW.block<3, 3>(3, 3) = R_EW.block<3, 3>(0, 0);

  // Rotate the velocity into E frame.
  Eigen::MatrixXd J_WE_E =
      R_EW *
      robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(cache, frame_E);

  for (int i = 0; i < 6; i++) {
    J_WE_E.row(i) = gain_E(i) * J_WE_E.row(i);
  }

  Eigen::Vector6d V_WE_E = R_EW * V_WE;
  V_WE_E = (V_WE_E.array() * gain_E.array()).matrix();

  Eigen::Vector6d V_WE_E_dir = V_WE_E.normalized();
  double V_WE_E_mag = V_WE_E.norm();

  Eigen::MatrixXd A(6, J_WE_E.cols() + 1);
  A.topLeftCorner(6, J_WE_E.cols()) = J_WE_E;
  A.topRightCorner(6, 1) = -V_WE_E_dir;
  prog.AddLinearEqualityConstraint(A, Eigen::Vector6d::Zero(), {v, alpha});
  auto err_cost = prog.AddQuadraticErrorCost(
      drake::Vector1<double>(1), drake::Vector1<double>(V_WE_E_mag), alpha);

  /*
  prog.AddL2NormCost(J_WE_E, V_WE_E, v);
  */

  // Add a small regularization
  prog.AddQuadraticCost(1e-3 * identity_ * dt * dt,
                        1e-3 * (cache.getQ() - q_nominal) * dt,
                        1e-3 * (cache.getQ() - q_nominal).squaredNorm(), v);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_WE_E, Eigen::ComputeFullV);

  // Add v constraint
  prog.AddBoundingBoxConstraint(v_lower_, v_upper_, v);

  // Add constrained the unconstrained dof's velocity to be small, which is used
  // to fullfil the regularization cost.
  prog.AddLinearConstraint(svd.matrixV().col(6).transpose(),
                           -unconstrained_dof_v_limit_,
                           unconstrained_dof_v_limit_, v);

  // Add q upper and lower joint limit.
  prog.AddLinearConstraint(identity_ * dt, q_lower_ - cache.getQ(),
                           q_upper_ - cache.getQ(), v);

  // Do the collision constraints.
  for (const std::pair<Capsule, Capsule>& col_pair : collisions) {
    Eigen::Vector3d p0, p1;
    const Capsule& c0 = col_pair.first;
    const Capsule& c1 = col_pair.second;

    // Computes the closest points in world frame.
    c0.GetClosestPointsOnAxis(c1, &p0, &p1);
    // Transform p0 and p1 into their respective body frames.
    Eigen::Isometry3d X0(Eigen::Translation3d(robot_->CalcBodyPoseInWorldFrame(cache, c0.get_body()).inverse() * p0));
    Eigen::Isometry3d X1(Eigen::Translation3d(robot_->CalcBodyPoseInWorldFrame(cache, c1.get_body()).inverse() * p1));

    auto J0 = robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(
        cache, c0.get_body(), X0);
    auto J1 = robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(
        cache, c1.get_body(), X1);

    Eigen::Matrix3d Rinv = c0.ComputeEscapeFrame(p0, p1).transpose();

    auto AA = Rinv * (J1.bottomRows(3) - J0.bottomRows(3)) * dt;
    Eigen::Vector3d bb = Rinv * (p1 - p0);
    drake::Vector1<double> min_dist(-(bb[2] - c0.get_radius() - c1.get_radius()));
    drake::Vector1<double> max_dist(1e6);
    prog.AddLinearConstraint(AA.row(2), min_dist, max_dist, v);
  }

  // Solve
  drake::solvers::SolutionResult result = prog.Solve();
  DRAKE_DEMAND(result == drake::solvers::SolutionResult::kSolutionFound);
  ret = prog.GetSolution(v);

  Eigen::VectorXd cost(1);
  err_cost.constraint()->Eval(prog.GetSolution(alpha), cost);
  // Not tracking the desired vel norm, and computed vel is small.
  *is_stuck = cost(0) > 5 && prog.GetSolution(alpha).norm() < 1e-3;
  // std::cout << "cost: " << cost(0) << ", " << prog.GetSolution(alpha).norm() << "\n";

  /*
  *is_stuck = false;
  */

  return ret;
}

bool JacobianIk::Plan(const Eigen::VectorXd &q0,
                      const std::vector<double> &times,
                      const std::vector<Eigen::Isometry3d> &pose_traj,
                      const RigidBodyFrame<double> &frame_E,
                      const Eigen::VectorXd &q_nominal,
                      std::vector<Eigen::VectorXd> *q_sol) const {
  DRAKE_DEMAND(times.size() == pose_traj.size());

  KinematicsCache<double> cache = robot_->CreateKinematicsCache();
  Eigen::VectorXd q_now = q0;
  Eigen::Isometry3d pose_now;
  double time_now = 0;

  Eigen::VectorXd v;
  bool is_stuck;

  q_sol->resize(pose_traj.size());

  for (size_t t = 0; t < pose_traj.size(); ++t) {
    cache.initialize(q_now);
    robot_->doKinematics(cache);

    pose_now = robot_->CalcFramePoseInWorldFrame(cache, frame_E);
    double dt = times[t] - time_now;
    Eigen::Vector6d V_WE_d =
        ComputePoseDiffInWorldFrame(pose_now, pose_traj[t]) / dt;

    v = ComputeDofVelocity(cache, {}, frame_E, V_WE_d, q_nominal, dt, &is_stuck);

    q_now += v * dt;
    time_now = times[t];
    (*q_sol)[t] = q_now;
  }

  return false;
}

} // namespace robot_bridge
