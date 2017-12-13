#include "robot_bridge/robot_bridge_common.h"

#include "drake/multibody/ik_options.h"
#include "drake/multibody/rigid_body_ik.h"

#include "drake/util/drakeUtil.h"

namespace robot_bridge {

const std::string kEEName("iiwa_link_7");

RobotState::RobotState(const RigidBodyTree<double> *iiwa,
                       const RigidBodyFrame<double> *frame_T)
    : iiwa_(iiwa), frame_T_(frame_T), cache_(iiwa_->CreateKinematicsCache()),
      q_(Eigen::VectorXd::Zero(iiwa_->get_num_positions())),
      v_(Eigen::VectorXd::Zero(iiwa_->get_num_velocities())),
      trq_(Eigen::VectorXd::Zero(iiwa_->get_num_actuators())),
      ext_trq_(Eigen::VectorXd::Zero(iiwa_->get_num_actuators())) {
  DRAKE_DEMAND(iiwa_->get_num_positions() == iiwa_->get_num_velocities());
  DRAKE_DEMAND(iiwa_->get_num_actuators() == iiwa_->get_num_velocities());
}

bool RobotState::UpdateState(const drake::lcmt_iiwa_status &msg) {
  // Check msg.
  if (msg.num_joints != iiwa_->get_num_positions()) {
    std::cout << "msg joints: " << msg.num_joints << std::endl;
  }
  DRAKE_DEMAND(msg.num_joints == iiwa_->get_num_positions());

  const double cur_time = msg.utime / 1e6;
  // Same time stamp, should just return.
  if (init_ && cur_time == time_)
    return false;

  if (init_) {
    delta_time_ = cur_time - time_;
  } else {
    delta_time_ = 0;
  }

  // Update time, position, and torque.
  time_ = cur_time;
  for (int i = 0; i < msg.num_joints; ++i) {
    q_[i] = msg.joint_position_measured[i];
    v_[i] = msg.joint_velocity_estimated[i];
    trq_[i] = msg.joint_torque_measured[i];
    ext_trq_[i] = msg.joint_torque_external[i];
  }

  // Update kinematics.
  cache_.initialize(q_, v_);
  iiwa_->doKinematics(cache_);
  J_ = iiwa_->CalcFrameSpatialVelocityJacobianInWorldFrame(cache_, *frame_T_);

  // ext_trq = trq_measured - trq_id
  //         = M * qdd + h - J^T * F - (M * qdd + h)
  //         = -J^T * F
  // I think the measured trq_ext is the flip side.
  ext_wrench_ = J_.transpose().colPivHouseholderQr().solve(ext_trq_);
  // ext_wrench_ = J_.transpose().colPivHouseholderQr().solve(-ext_trq_);

  X_WT_ = iiwa_->CalcFramePoseInWorldFrame(cache_, *frame_T_);
  V_WT_ = iiwa_->CalcFrameSpatialVelocityInWorldFrame(cache_, *frame_T_);

  init_ = true;

  return true;
}

Eigen::Isometry3d RobotState::get_X_WP(const RigidBodyFrame<double> &P) const {
  if (!init_) {
    throw std::runtime_error("Unitialized state.");
  }
  return iiwa_->CalcFramePoseInWorldFrame(cache_, P);
}

Eigen::Vector6d RobotState::get_V_WP(const RigidBodyFrame<double> &P) const {
  if (!init_) {
    throw std::runtime_error("Unitialized state.");
  }
  return iiwa_->CalcFrameSpatialVelocityInWorldFrame(cache_, P);
}

Eigen::Matrix<double, 7, 1> pose_to_vec(const Eigen::Isometry3d &pose) {
  Eigen::Matrix<double, 7, 1> ret;
  ret.head<3>() = pose.translation();

  Eigen::Quaterniond quat(pose.linear());
  ret[3] = quat.w();
  ret[4] = quat.x();
  ret[5] = quat.y();
  ret[6] = quat.z();

  return ret;
}

Eigen::VectorXd PointIk(const Eigen::Isometry3d &X_WT,
                        const RigidBodyFrame<double> &frame_T,
                        const Eigen::VectorXd &q_ini,
                        RigidBodyTree<double> *robot) {
  std::cout << "PointIk: X_WT:\n" << X_WT.matrix() << "\n\n";

  std::vector<RigidBodyConstraint *> constraint_array;

  IKoptions ikoptions(robot);

  Eigen::Isometry3d X_WE = X_WT * frame_T.get_transform_to_body().inverse();

  Eigen::Vector3d pos_tol(0.001, 0.001, 0.001);
  double rot_tol = 0.001;
  Eigen::Vector3d pos_lb = X_WE.translation() - pos_tol;
  Eigen::Vector3d pos_ub = X_WE.translation() + pos_tol;

  WorldPositionConstraint pos_con(
      robot, frame_T.get_rigid_body().get_body_index(), Eigen::Vector3d::Zero(),
      pos_lb, pos_ub, Eigen::Vector2d::Zero());

  constraint_array.push_back(&pos_con);

  // Adds a rotation constraint.
  WorldQuatConstraint quat_con(robot, frame_T.get_rigid_body().get_body_index(),
                               drake::math::rotmat2quat(X_WE.linear()), rot_tol,
                               Eigen::Vector2d::Zero());
  constraint_array.push_back(&quat_con);

  Eigen::VectorXd q_res = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);

  int info;
  std::vector<std::string> infeasible_constraints;
  inverseKin(robot, q_ini, zero, constraint_array.size(),
             constraint_array.data(), ikoptions, &q_res, &info,
             &infeasible_constraints);

  DRAKE_DEMAND(info == 1);
  return q_res;
}

static Eigen::Matrix3d FlatYAxisFrame(const Eigen::Vector3d &z) {
  const Eigen::Vector3d py_z = z.normalized();
  const Eigen::Vector3d py_z_proj = Eigen::Vector3d(z(0), z(1), 0).normalized();
  if (py_z_proj.norm() < 1e-3)
    return Eigen::Matrix3d::Identity();

  const Eigen::Vector3d py_y =
      py_z_proj.cross(Eigen::Vector3d::UnitZ()).normalized();
  const Eigen::Vector3d py_x = py_y.cross(py_z).normalized();
  Eigen::Matrix3d rot;
  rot.col(0) = py_x;
  rot.col(1) = py_y;
  rot.col(2) = py_z;
  return rot;
}

Eigen::VectorXd GazeIk(const Eigen::Vector3d &target_in_world,
                       const Eigen::Vector3d &camera_in_world,
                       const RigidBodyFrame<double> &frame_C,
                       const Eigen::VectorXd &q_ini,
                       RigidBodyTree<double> *robot) {
  std::vector<RigidBodyConstraint *> constraint_array;

  IKoptions ikoptions(robot);

  const Eigen::Isometry3d &X_BC = frame_C.get_transform_to_body();
  const int body_idx = frame_C.get_rigid_body().get_body_index();

  // Gaze dir constraint.
  const Eigen::Vector3d gaze_ray_dir_in_body = X_BC.linear().col(2);
  const Eigen::Vector3d gaze_ray_origin_in_body = X_BC.translation();

  WorldGazeTargetConstraint con(robot, body_idx, gaze_ray_dir_in_body,
                                target_in_world, gaze_ray_origin_in_body, 0.01,
                                Eigen::Vector2d::Zero());

  constraint_array.push_back(&con);

  // Camera position constraint.
  Eigen::Vector3d p_WB =
      (Eigen::Translation<double, 3>(camera_in_world) * X_BC.inverse())
          .translation();
  Eigen::Vector3d pos_tol(0.001, 0.001, 0.001);
  Eigen::Vector3d pos_lb = p_WB - pos_tol;
  Eigen::Vector3d pos_ub = p_WB + pos_tol;

  WorldPositionConstraint pos_con(robot, body_idx, Eigen::Vector3d::Zero(),
                                  pos_lb, pos_ub, Eigen::Vector2d::Zero());

  constraint_array.push_back(&pos_con);

  Eigen::VectorXd q_res = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);

  int info;
  std::vector<std::string> infeasible_constraints;
  inverseKin(robot, q_ini, zero, constraint_array.size(),
             constraint_array.data(), ikoptions, &q_res, &info,
             &infeasible_constraints);

  DRAKE_DEMAND(info == 1);
  return q_res;
}

Eigen::VectorXd GazeIk2(const Eigen::Vector3d &target_in_world,
                        const Eigen::Vector3d &target_to_camera_in_world,
                        double min_dist, double max_dist,
                        const RigidBodyFrame<double> &frame_C,
                        const Eigen::VectorXd &q_ini,
                        RigidBodyTree<double> *robot) {
  std::vector<RigidBodyConstraint *> constraint_array;

  IKoptions ikoptions(robot);

  const Eigen::Isometry3d &X_BC = frame_C.get_transform_to_body();
  const int body_idx = frame_C.get_rigid_body().get_body_index();

  // Gaze dir constraint.
  const Eigen::Vector3d gaze_ray_dir_in_body = X_BC.linear().col(2);
  const Eigen::Vector3d gaze_ray_origin_in_body = X_BC.translation();

  WorldGazeTargetConstraint con(robot, body_idx, gaze_ray_dir_in_body,
                                target_in_world, gaze_ray_origin_in_body, 0.01,
                                Eigen::Vector2d::Zero());

  constraint_array.push_back(&con);

  // Camera position constraint.
  Eigen::Isometry3d X_WTgt = Eigen::Isometry3d::Identity();
  X_WTgt.translation() = target_in_world;
  X_WTgt.linear() = FlatYAxisFrame(target_to_camera_in_world);

  Eigen::Vector3d pos_lb(0, 0, min_dist);
  Eigen::Vector3d pos_ub(0, 0, max_dist);

  WorldPositionInFrameConstraint pos_con(robot, body_idx, X_BC.translation(),
                                         X_WTgt.matrix(), pos_lb, pos_ub,
                                         Eigen::Vector2d::Zero());

  constraint_array.push_back(&pos_con);

  Eigen::VectorXd q_res = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);

  int info;
  std::vector<std::string> infeasible_constraints;
  inverseKin(robot, q_ini, zero, constraint_array.size(),
             constraint_array.data(), ikoptions, &q_res, &info,
             &infeasible_constraints);

  DRAKE_DEMAND(info == 1);
  return q_res;
}

std::vector<Eigen::VectorXd> ComputeCalibrationConfigurations(
    const RigidBodyTree<double> &robot, const RigidBodyFrame<double> &frame_C,
    const Eigen::VectorXd &q0, const Eigen::Vector3d &p_WP, double min_dist,
    double width, double height, int num_width_pt, int num_height_pt) {
  KinematicsCache<double> cache = robot.CreateKinematicsCache();
  cache.initialize(q0);
  robot.doKinematics(cache);

  const Eigen::Isometry3d X_WC0 =
      robot.CalcFramePoseInWorldFrame(cache, frame_C);
  const Eigen::Vector3d C0_to_P = p_WP - X_WC0.translation();
  const double pyramid_height = C0_to_P.norm();

  Eigen::Isometry3d X_WP = Eigen::Isometry3d::Identity();
  X_WP.linear() = FlatYAxisFrame(C0_to_P);
  X_WP.translation() = p_WP;

  double dw = width / (num_width_pt - 1);
  double dh = height / (num_height_pt - 1);

  std::vector<Eigen::VectorXd> ret;
  for (int i = 0; i < num_width_pt; i++) {
    for (int j = 0; j < num_height_pt; j++) {
      Eigen::Vector3d p_PC(-height / 2. + j * dh, -width / 2. + i * dw,
                           -pyramid_height);
      p_PC = p_PC.normalized(); // * pyramid_height;
      Eigen::Vector3d p_WC = X_WP * p_PC;
      // ret.push_back(GazeIk(p_WP, p_WC, frame_C, q0,
      // (RigidBodyTree<double>*)&robot));
      ret.push_back(GazeIk2(p_WP, p_WC - p_WP, min_dist,
                            std::numeric_limits<double>::infinity(), frame_C,
                            q0, (RigidBodyTree<double> *)&robot));
    }
  }

  return ret;
}

std::vector<Eigen::VectorXd>
ScanAroundPoint(const RigidBodyTree<double> &robot,
                const RigidBodyFrame<double> &frame_C,
                const Eigen::Vector3d &p_WP, const Eigen::Vector3d &normal_W,
                double min_dist, double max_dist, double width, double height,
                int dw, int dh) {
  Eigen::Isometry3d X_WP = Eigen::Isometry3d::Identity();
  X_WP.linear() = FlatYAxisFrame(normal_W.normalized());
  X_WP.translation() = p_WP;
  std::cout << X_WP.matrix() << "\n";

  std::vector<RigidBodyConstraint *> constraint_array;
  std::vector<std::unique_ptr<RigidBodyConstraint>> constraint_array_real;

  int ctr = 0;
  double dt = 3;
  std::vector<double> Times;

  RigidBodyTree<double> *robot_ptr = (RigidBodyTree<double> *)&robot;
  /*
  std::vector<Eigen::Vector3d> corners(4);
  corners[0] = Eigen::Vector3d(-height / 2., -width / 2., 1);
  corners[1] = Eigen::Vector3d(-height / 2., width / 2., 1);
  corners[2] = Eigen::Vector3d(height / 2., width / 2., 1);
  corners[3] = Eigen::Vector3d(height / 2., -width / 2., 1);
  */
  std::vector<Eigen::Vector3d> corners;
  double dy = width / (dw - 1);
  double dx = height / (dh - 1);
  for (int j = 0; j < dh; j++) {
    for (int i = 0; i < dw; i++) {
      double x = -height / 2. + j * dx;
      double y = -width / 2. + i * dy;
      corners.push_back(Eigen::Vector3d(x, y, 1));
    }
  }

  for (auto p_PC : corners) {
    double t_now = ctr * dt;

    p_PC = p_PC.normalized();
    Eigen::Vector3d p_WC = X_WP * p_PC;

    const Eigen::Isometry3d &X_BC = frame_C.get_transform_to_body();
    const int body_idx = frame_C.get_rigid_body().get_body_index();

    // Gaze dir constraint.
    const Eigen::Vector3d gaze_ray_dir_in_body = X_BC.linear().col(2);
    const Eigen::Vector3d gaze_ray_origin_in_body = X_BC.translation();

    constraint_array_real.emplace_back(
        std::make_unique<WorldGazeTargetConstraint>(
            robot_ptr, body_idx, gaze_ray_dir_in_body, p_WP,
            gaze_ray_origin_in_body, 0.05, Eigen::Vector2d(t_now, t_now)));
    constraint_array.push_back(constraint_array_real.back().get());

    // Camera position constraint.
    Eigen::Isometry3d X_WTgt = Eigen::Isometry3d::Identity();
    X_WTgt.translation() = p_WP;
    X_WTgt.linear() = FlatYAxisFrame(p_WC - p_WP);

    Eigen::Vector3d pos_lb(-0.02, -0.02, min_dist);
    Eigen::Vector3d pos_ub(0.02, 0.02, max_dist);

    constraint_array_real.emplace_back(
        std::make_unique<WorldPositionInFrameConstraint>(
            robot_ptr, body_idx, X_BC.translation(), X_WTgt.matrix(), pos_lb,
            pos_ub, Eigen::Vector2d(t_now, t_now)));
    constraint_array.push_back(constraint_array_real.back().get());

    ctr++;
    Times.push_back(t_now);
  }

  Eigen::VectorXd q_res = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(7);

  Eigen::VectorXd times(Times.size());
  Eigen::MatrixXd q0(robot.get_num_positions(), times.size());
  for (int i = 0; i < times.size(); i++) {
    q0.col(i) = robot.getZeroConfiguration();
    times[i] = Times[i];
  }

  IKoptions ikoptions(robot_ptr);
  ikoptions.setFixInitialState(false);
  auto result = inverseKinTrajSimple(robot_ptr, times, q0, q0, constraint_array,
                                     ikoptions);

  for (const auto &q : result.q_sol)
    std::cout << q.transpose() << "\n";

  for (const auto &info : result.info)
    DRAKE_DEMAND(info == 1);

  return result.q_sol;
}

// Assuming 3rd order
static bool CheckVelocityAndAccConstraints3(const Polynomial<double> &poly,
                                            double time, double v_lower,
                                            double v_upper, double vd_lower,
                                            double vd_upper) {
  Polynomial<double> v = poly.Derivative();
  Polynomial<double> vd = v.Derivative();

  DRAKE_DEMAND(poly.GetDegree() == 3);

  // Check vd.
  if (vd.EvaluateUnivariate(0) < vd_lower ||
      vd.EvaluateUnivariate(0) > vd_upper ||
      vd.EvaluateUnivariate(time) < vd_lower ||
      vd.EvaluateUnivariate(time) > vd_upper) {
    return false;
  }

  // Check v.
  if (v.EvaluateUnivariate(0) < v_lower || v.EvaluateUnivariate(0) > v_upper ||
      v.EvaluateUnivariate(time) < v_lower ||
      v.EvaluateUnivariate(time) > v_upper) {
    return false;
  }
  Eigen::VectorXd coeffs = poly.GetCoefficients();
  double extrema_t = -coeffs(2) / (3. * coeffs(3));
  if (extrema_t > 0 && extrema_t < time) {
    if (v.EvaluateUnivariate(extrema_t) < v_lower ||
        v.EvaluateUnivariate(extrema_t) > v_upper) {
      return false;
    }
  }

  return true;
}

static bool CheckTrajVelAndAccConstraints(
    const PiecewisePolynomial<double> &traj, const Eigen::MatrixXd &v_lower,
    const Eigen::MatrixXd &v_upper, const Eigen::MatrixXd &vd_lower,
    const Eigen::MatrixXd &vd_upper) {
  int num_rows = v_lower.rows();
  int num_cols = v_lower.cols();
  for (int t = 0; t < traj.getNumberOfSegments(); t++) {
    for (int r = 0; r < num_rows; r++) {
      for (int c = 0; c < num_cols; c++) {
        if (!CheckVelocityAndAccConstraints3(
                traj.getPolynomial(t, r, c), traj.getDuration(t), v_lower(r, c),
                v_upper(r, c), vd_lower(r, c), vd_upper(r, c))) {
          return false;
        }
      }
    }
  }

  return true;
}

static PiecewisePolynomial<double>
GuessTrajTime3(const std::vector<Eigen::MatrixXd> &q, const Eigen::MatrixXd &v0,
               const Eigen::MatrixXd &v1, const Eigen::MatrixXd &v_lower,
               const Eigen::MatrixXd &v_upper, const Eigen::MatrixXd &vd_lower,
               const Eigen::MatrixXd &vd_upper) {
  std::vector<double> times(q.size(), 0);
  std::vector<double> dt(q.size() - 1, 1);

  for (size_t t = 1; t < q.size(); t++) {
    times[t] = times[t - 1] + dt[t - 1];
  }

  PiecewisePolynomial<double> traj =
      PiecewisePolynomial<double>::Cubic(times, q, v0, v1);

  // Check constraints, and adjust dt.
  int num_rows = v0.rows();
  int num_cols = v0.cols();

  while (true) {
    bool all_ok = true;
    for (size_t t = 0; t < q.size() - 1; t++) {
      bool ok = true;
      for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
          ok &= CheckVelocityAndAccConstraints3(
              traj.getPolynomial(t, r, c), dt[t], v_lower(r, c), v_upper(r, c),
              vd_lower(r, c), vd_upper(r, c));
        }
      }
      if (!ok) {
        dt[t] = 1.5 * dt[t];
        all_ok = false;
      }
    }

    if (all_ok)
      break;

    for (size_t t = 1; t < q.size(); t++) {
      times[t] = times[t - 1] + dt[t - 1];
    }
    traj = PiecewisePolynomial<double>::Cubic(times, q, v0, v1);
  }

  return traj;
}

static PiecewisePolynomial<double> LineSearchSingleCubicSpline(
    const std::vector<Eigen::MatrixXd> &q, double min_time, double max_time,
    const Eigen::MatrixXd &v0, const Eigen::MatrixXd &v1,
    const Eigen::MatrixXd &v_lower, const Eigen::MatrixXd &v_upper,
    const Eigen::MatrixXd &vd_lower, const Eigen::MatrixXd &vd_upper) {
  DRAKE_DEMAND(q.size() == 2);

  double mid_time = (min_time + max_time) / 2.;
  std::vector<double> times = {0, mid_time};
  PiecewisePolynomial<double> traj =
      PiecewisePolynomial<double>::Cubic(times, q, v0, v1);

  if (CheckTrajVelAndAccConstraints(traj, v_lower, v_upper, vd_lower,
                                    vd_upper)) {
    if (std::fabs(mid_time - min_time) < 1e-2) {
      return traj;
    } else {
      return LineSearchSingleCubicSpline(q, min_time, mid_time, v0, v1, v_lower,
                                         v_upper, vd_lower, vd_upper);
    }
  } else {
    return LineSearchSingleCubicSpline(q, mid_time, max_time, v0, v1, v_lower,
                                       v_upper, vd_lower, vd_upper);
  }
}

PiecewisePolynomial<double> LineSearchSingleCubicSpline(
    const std::vector<Eigen::MatrixXd> &q, const Eigen::MatrixXd &v0,
    const Eigen::MatrixXd &v1, const Eigen::MatrixXd &v_lower,
    const Eigen::MatrixXd &v_upper, const Eigen::MatrixXd &vd_lower,
    const Eigen::MatrixXd &vd_upper) {
  DRAKE_DEMAND(q.size() == 2);

  PiecewisePolynomial<double> max_time_traj =
      GuessTrajTime3(q, v0, v1, v_lower, v_upper, vd_lower, vd_upper);

  return LineSearchSingleCubicSpline(q, 0.1, max_time_traj.getEndTime(), v0, v1,
                                     v_lower, v_upper, vd_lower, vd_upper);
}

PiecewisePolynomial<double>
RetimeTrajCubic(const std::vector<Eigen::MatrixXd> &q,
                const Eigen::MatrixXd &v0, const Eigen::MatrixXd &v1,
                const Eigen::MatrixXd &v_lower, const Eigen::MatrixXd &v_upper,
                const Eigen::MatrixXd &vd_lower,
                const Eigen::MatrixXd &vd_upper) {
  if (q.size() == 2) {
    return LineSearchSingleCubicSpline(q, v0, v1, v_lower, v_upper, vd_lower,
                                       vd_upper);
  } else {
    return GuessTrajTime3(q, v0, v1, v_lower, v_upper, vd_lower, vd_upper);
  }
}

} // namespace robot_bridge
