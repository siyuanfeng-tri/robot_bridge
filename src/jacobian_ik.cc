#include "robot_bridge/jacobian_ik.h"
#include "robot_bridge/util.h"

#include <memory>

#include "drake/common/text_logging.h"
#include "drake/multibody/ik_options.h"
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
  const int num_joints = robot_->get_num_velocities();

  q_lower_ = robot_->joint_limit_min;
  q_upper_ = robot_->joint_limit_max;
  v_lower_ = Eigen::VectorXd::Constant(num_joints, -2);
  v_upper_ = Eigen::VectorXd::Constant(num_joints, 2);
  vd_lower_ = Eigen::VectorXd::Constant(num_joints, -40);
  vd_upper_ = Eigen::VectorXd::Constant(num_joints, 40);
  unconstrained_dof_v_limit_ = Eigen::VectorXd::Constant(1, 0.6);

  identity_ = Eigen::MatrixXd::Identity(num_joints, num_joints);
  zero_ = Eigen::VectorXd::Zero(num_joints);

  DRAKE_DEMAND((q_upper_.array() >= q_lower_.array()).all());
}

void JacobianIk::SetJointSpeedLimit(const Eigen::VectorXd &v_upper,
                                    const Eigen::VectorXd &v_lower) {
  DRAKE_DEMAND(v_upper.size() == v_lower.size());
  DRAKE_DEMAND(v_lower.size() == v_lower_.size());
  v_lower_ = v_lower;
  v_upper_ = v_upper;

  DRAKE_DEMAND((v_upper_.array() >= v_lower_.array()).all());
}

void JacobianIk::SetJointAccelerationLimit(const Eigen::VectorXd &vd_upper,
                                           const Eigen::VectorXd &vd_lower) {
  DRAKE_DEMAND(vd_upper.size() == vd_lower.size());
  DRAKE_DEMAND(vd_lower.size() == vd_lower_.size());
  vd_lower_ = vd_lower;
  vd_upper_ = vd_upper;

  DRAKE_DEMAND((vd_upper_.array() >= vd_lower_.array()).all());
}

JacobianIk::JacobianIk(const RigidBodyTree<double> *robot) : robot_{robot} {
  Setup();
}

Eigen::VectorXd JacobianIk::ComputeDofVelocity(
    const KinematicsCache<double> &cache,
    const std::vector<std::pair<Capsule, Capsule>> &collisions,
    const RigidBodyFrame<double> &frame_E, const Eigen::Vector6d &V_WE,
    double dt, const Eigen::VectorXd &q_nominal, const Eigen::VectorXd &v_last,
    bool *is_stuck, const Eigen::Vector6d &gain_E) const {
  DRAKE_DEMAND(q_nominal.size() == robot_->get_num_positions());
  DRAKE_DEMAND(dt > 0);

  Eigen::VectorXd ret;

  drake::solvers::MathematicalProgram prog;
  drake::solvers::VectorXDecisionVariable v =
      prog.NewContinuousVariables(robot_->get_num_velocities(), "v");
  drake::solvers::VectorXDecisionVariable alpha =
      prog.NewContinuousVariables(1, "alpha");

  Eigen::Isometry3d X_WE = robot_->CalcFramePoseInWorldFrame(cache, frame_E);

  // Rotate the world velocity into E frame.
  drake::Matrix6<double> R_EW = drake::Matrix6<double>::Zero();
  R_EW.block<3, 3>(0, 0) = X_WE.linear().transpose();
  R_EW.block<3, 3>(3, 3) = R_EW.block<3, 3>(0, 0);

  Eigen::MatrixXd J_WE_E_6d =
      R_EW *
      robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(cache, frame_E);
  Eigen::Vector6d V_WE_E_6d = R_EW * V_WE;

  // Pick the constrained motions.
  int num_cart_constraints = 0;
  for (int i = 0; i < 6; i++) {
    if (gain_E(i) > 0) {
      J_WE_E_6d.row(num_cart_constraints) = gain_E(i) * J_WE_E_6d.row(i);
      V_WE_E_6d(num_cart_constraints) = gain_E(i) * V_WE_E_6d(i);
      num_cart_constraints++;
    }
  }
  Eigen::MatrixXd J_WE_E = J_WE_E_6d.topRows(num_cart_constraints);
  Eigen::VectorXd V_WE_E = V_WE_E_6d.head(num_cart_constraints);

  Eigen::VectorXd V_WE_E_dir = V_WE_E.normalized();
  double V_WE_E_mag = V_WE_E.norm();

  // Constrain the end effector motion to be in the direction of V_WE_E_dir,
  // and penalize magnitude difference from V_WE_E_mag.
  Eigen::MatrixXd A(num_cart_constraints, J_WE_E.cols() + 1);
  A.topLeftCorner(num_cart_constraints, J_WE_E.cols()) = J_WE_E;
  A.topRightCorner(num_cart_constraints, 1) = -V_WE_E_dir;
  prog.AddLinearEqualityConstraint(
      A, Eigen::VectorXd::Zero(num_cart_constraints), {v, alpha});
  auto err_cost = prog.AddQuadraticErrorCost(
      drake::Vector1<double>(1), drake::Vector1<double>(V_WE_E_mag), alpha);

  // Add a small regularization.
  auto posture_cost = prog.AddQuadraticCost(
      1e-3 * identity_ * dt * dt, 1e-3 * (cache.getQ() - q_nominal) * dt,
      1e-3 * (cache.getQ() - q_nominal).squaredNorm(), v);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_WE_E, Eigen::ComputeFullV);

  // Add v constraint.
  prog.AddBoundingBoxConstraint(v_lower_, v_upper_, v);

  // Add vd constraint.
  prog.AddLinearConstraint(identity_,
                           vd_lower_ * dt + v_last,
                           vd_upper_ * dt + v_last,
                           v);

  // Add constrained the unconstrained dof's velocity to be small, which is used
  // to fullfil the regularization cost.
  for (int i = num_cart_constraints; i < svd.matrixV().cols(); i++) {
    prog.AddLinearConstraint(svd.matrixV().col(i).transpose(),
                             -unconstrained_dof_v_limit_,
                             unconstrained_dof_v_limit_, v);
  }

  // Add q upper and lower joint limit.
  prog.AddLinearConstraint(identity_ * dt, q_lower_ - cache.getQ(),
                           q_upper_ - cache.getQ(), v);

  // Do the collision constraints.
  for (const std::pair<Capsule, Capsule> &col_pair : collisions) {
    Eigen::Vector3d p0, p1;
    const Capsule &c0 = col_pair.first;
    const Capsule &c1 = col_pair.second;

    // Computes the closest points in world frame.
    c0.GetClosestPointsOnAxis(c1, &p0, &p1);
    // Transform p0 and p1 into their respective body frames.
    Eigen::Isometry3d X0(Eigen::Translation3d(
        robot_->CalcBodyPoseInWorldFrame(cache, c0.get_body()).inverse() * p0));
    Eigen::Isometry3d X1(Eigen::Translation3d(
        robot_->CalcBodyPoseInWorldFrame(cache, c1.get_body()).inverse() * p1));

    auto J0 = robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(
        cache, c0.get_body(), X0);
    auto J1 = robot_->CalcFrameSpatialVelocityJacobianInWorldFrame(
        cache, c1.get_body(), X1);

    Eigen::Matrix3d Rinv = c0.ComputeEscapeFrame(p0, p1).transpose();

    auto AA = Rinv * (J1.bottomRows(3) - J0.bottomRows(3)) * dt;
    Eigen::Vector3d bb = Rinv * (p1 - p0);
    drake::Vector1<double> min_dist(
        -(bb[2] - c0.get_radius() - c1.get_radius()));
    drake::Vector1<double> max_dist(1e6);
    prog.AddLinearConstraint(AA.row(2), min_dist, max_dist, v);
  }

  // Solve
  drake::solvers::SolutionResult result = prog.Solve();
  if (result != drake::solvers::SolutionResult::kSolutionFound) {
    std::cout << "SCS CANT SOLVE: " << result << "\n";
    *is_stuck = false;
    return Eigen::VectorXd::Zero(robot_->get_num_velocities());
  }
  ret = prog.GetSolution(v);

  Eigen::VectorXd cost(1);
  err_cost.constraint()->Eval(prog.GetSolution(alpha), cost);
  // Not tracking the desired vel norm, and computed vel is small.
  *is_stuck = cost(0) > 5 && prog.GetSolution(alpha)[0] <= 1e-2;

  // std::cout << "err_cost: " << cost(0) << ", " <<
  // prog.GetSolution(alpha).norm() << "\n";

  posture_cost.constraint()->Eval(prog.GetSolution(v), cost);
  // std::cout << "posture_cost: " << cost(0) << "\n";

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

  Eigen::VectorXd v = Eigen::VectorXd::Zero(q_nominal.size());
  bool is_stuck;

  q_sol->resize(pose_traj.size());

  for (size_t t = 0; t < pose_traj.size(); ++t) {
    cache.initialize(q_now);
    robot_->doKinematics(cache);

    pose_now = robot_->CalcFramePoseInWorldFrame(cache, frame_E);
    double dt = times[t] - time_now;
    Eigen::Vector6d V_WE_d =
        ComputePoseDiffInWorldFrame(pose_now, pose_traj[t]) / dt;

    v = ComputeDofVelocity(cache, {}, frame_E, V_WE_d, dt, q_nominal, v,
                           &is_stuck);

    q_now += v * dt;
    time_now = times[t];
    (*q_sol)[t] = q_now;
  }

  return false;
}

} // namespace robot_bridge
