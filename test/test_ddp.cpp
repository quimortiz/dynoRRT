#include "dynoRRT/eigen_conversions.hpp"
#include <Eigen/src/Core/Matrix.h>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "dynoRRT/dynorrt_macros.h"
#include "dynoRRT/eigen_conversions.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;
using std::vector;
using namespace dynorrt;

// TODO
//  Use a generic implementation with fx, fu, lxx, luu, lux, l, lf
// Backtraking line search (with armijo condition?) Or just
// reduce alpha.
//
// Think about how to add constraints bounds?
// Or soft as cost.

void ddp(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd Q,
         Eigen::MatrixXd Qf, Eigen::MatrixXd R, vector<Eigen::VectorXd> xs,
         vector<Eigen::VectorXd> us) {

  CHECK_PRETTY_DYNORRT__(xs.size() == us.size() + 1);
  int N = us.size();
  int nx = xs[0].size();
  int nu = us[0].size();

  vector<Eigen::MatrixXd> fx(N);
  vector<Eigen::MatrixXd> fu(N);

  for (auto &f : fx) {
    f = A;
  }
  for (auto &f : fu) {
    f = B;
  }

  vector<Eigen::MatrixXd> lxxs(N + 1);
  vector<Eigen::MatrixXd> luus(N);
  vector<Eigen::MatrixXd> luxs(N);

  for (size_t i = 0; i < N; i++) {
    lxxs[i] = Q;
  }
  lxxs[N] = Qf;

  for (auto &luu : luus) {
    luu = R;
  }
  for (auto &lux : luxs) {
    lux = Eigen::MatrixXd::Zero(nu, nx);
  }

  vector<Eigen::VectorXd> lxs(N + 1);
  vector<Eigen::VectorXd> lus(N);
  for (size_t i = 0; i < N; i++) {
    lxs[i] = Q * xs[i];
  }
  lxs[N] = Qf * xs[N];
  for (size_t i = 0; i < N; i++) {
    lus[i] = R * us[i];
  }

  vector<Eigen::MatrixXd> vxxs(N + 1);
  vector<Eigen::VectorXd> vxs(N + 1);

  vector<Eigen::VectorXd> ks(N);
  vector<Eigen::MatrixXd> Ks(N);

  vector<Eigen::VectorXd> qxs(N);
  vector<Eigen::VectorXd> qus(N);
  vector<Eigen::MatrixXd> qxxs(N);
  vector<Eigen::MatrixXd> quus(N);
  vector<Eigen::MatrixXd> quxs(N);

  int max_num_it_ddp = 3;
  double alpha = 1; // between 0 and 1.

  // get the cost

  auto l = [&](const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    return (0.5 * x.transpose() * Q * x + 0.5 * u.transpose() * R * u).value();
  };
  auto lf = [&](const Eigen::VectorXd &x) {
    return (0.5 * x.transpose() * Qf * x).value();
  };

  double cost = 0;
  for (size_t i = 0; i < N; i++) {
    cost += l(xs[i], us[i]);
  }
  cost += lf(xs[N]);
  std::cout << "initial cost: " << cost << std::endl;

  for (size_t j = 0; j < max_num_it_ddp; j++) {
    std::cout << "iteration: " << j << std::endl;

    // backward pass
    //

    for (size_t i = 0; i < N; i++) {
      lxs[i] = Q * xs[i];
    }
    lxs[N] = Qf * xs[N];
    for (size_t i = 0; i < N; i++) {
      lus[i] = R * us[i];
    }

    vxxs[N] = lxxs[N];
    vxs[N] = lxs[N];
    for (size_t i = 0; i < N; i++) {
      int idx = N - i - 1;
      qxs[idx] = lxs[idx] + fx[idx].transpose() * vxs[idx + 1];
      qus[idx] = lus[idx] + fu[idx].transpose() * vxs[idx + 1];
      qxxs[idx] = lxxs[idx] + fx[idx].transpose() * vxxs[idx + 1] * fx[idx];
      quus[idx] = luus[idx] + fu[idx].transpose() * vxxs[idx + 1] * fu[idx];
      quxs[idx] = luxs[idx] + fu[idx].transpose() * vxxs[idx + 1] * fx[idx];

      ks[idx] = -quus[idx].inverse() * qus[idx];
      Ks[idx] = -quus[idx].inverse() * quxs[idx];
      vxs[idx] = qxs[idx] + quxs[idx].transpose() * ks[idx];
      vxxs[idx] = qxxs[idx] + quxs[idx].transpose() * Ks[idx];
    }
    // forward pass

    // note: the first state is kept fixed

    vector<Eigen::VectorXd> xs_new = xs;
    vector<Eigen::VectorXd> us_new = us;

    auto f = [&](const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
      return A * x + B * u;
    };

    for (size_t idx = 0; idx < N; idx++) {
      us_new[idx] =
          us[idx] + alpha * ks[idx] + Ks[idx] * (xs_new[idx] - xs[idx]);
      xs_new[idx + 1] = f(xs_new[idx], us_new[idx]);
    }

    // check for convergence
    double max_diff = 0;
    for (size_t idx = 0; idx < N; idx++) {
      if (double dif =
              (xs_new[idx] - xs[idx]).norm() + (us_new[idx] - us[idx]).norm();
          dif > max_diff) {
        max_diff = dif;
      }
    }
    std::cout << "iteration: " << j << " max_diff: " << max_diff << std::endl;

    if (max_diff < 1e-5) {
      std::cout << "Converged after " << j << " iterations" << std::endl;
      break;
    }

    xs = xs_new;
    us = us_new;

    double cost = 0;
    for (size_t i = 0; i < N; i++) {
      cost += l(xs[i], us[i]);
    }
    cost += lf(xs[N]);
    std::cout << "cost: " << cost << std::endl;
  }

  std::ofstream myfile("/tmp/ddp.json");
  json j;
  j["xs"] = xs;
  j["us"] = us;
  myfile << j;
}

void ddp_example_1() {

  int N = 10;
  std::vector<Eigen::VectorXd> xs(N + 1);
  std::vector<Eigen::VectorXd> us(N);

  for (auto &x : xs) {
    x = Eigen::VectorXd::Zero(2);
    x(0) = 1;
    x(1) = .5;
  }
  for (auto &u : us) {
    u = Eigen::VectorXd::Zero(2);
  }

  // xnext = A*x + B*u
  Eigen::MatrixXd A(2, 2);
  A << 1, 0, 0, 1;
  Eigen::MatrixXd B(2, 2);
  B << 1, 0, 0, 1;

  // cost: . 5 xQx + .5 uRu
  Eigen::MatrixXd Q(2, 2);
  Q << 1, 0, 0, 1;
  Eigen::MatrixXd R(2, 2);
  R << 1, 0, 0, 1;

  Eigen::MatrixXd Qf(2, 2);
  Qf << 10, 0, 0, 10;

  ddp(A, B, Q, Qf, R, xs, us);
}

void ddp_example_2() {

  int N = 100;
  double dt = .1;
  std::vector<Eigen::VectorXd> xs(N + 1);
  std::vector<Eigen::VectorXd> us(N);

  for (auto &x : xs) {
    x = Eigen::VectorXd::Zero(2);
    x(0) = 1;
    x(1) = 0;
  }
  for (auto &u : us) {
    u = Eigen::VectorXd::Zero(1);
  }

  // double dt = .1;
  // xnext = A*x + B*u
  Eigen::MatrixXd A(2, 2);
  A << 1, dt, 0, 1;
  Eigen::MatrixXd B(2, 1);
  B << 0, dt;

  // cost: . 5 xQx + .5 uRu
  Eigen::MatrixXd Q(2, 2);
  // Q << 1, 0, 0, 1;
  Q << 0, 0, 0, 0;
  Eigen::MatrixXd R(1, 1);
  R << 1;

  Eigen::MatrixXd Qf(2, 2);
  Qf << 10, 0, 0, 10;

  ddp(A, B, Q, Qf, R, xs, us);
}

void ddp_example_3() {

  int N = 100;
  double dt = .1;
  std::vector<Eigen::VectorXd> xs(N + 1);
  std::vector<Eigen::VectorXd> us(N);

  for (auto &x : xs) {
    x = Eigen::VectorXd::Zero(4);
    x(0) = 1;
    x(1) = 0;
    x(2) = .5;
    x(3) = 1;
  }
  for (auto &u : us) {
    u = Eigen::VectorXd::Zero(2);
  }

  // double dt = .1;
  // xnext = A*x + B*u
  Eigen::MatrixXd A(4, 4);
  // A << 1, dt, 0, 1;

  A << 1, dt, 0, 0, //
      0, 1, 0, 0,   //
      0, 0, 1, dt,  //
      0, 0, 0, 1;

  Eigen::MatrixXd B(4, 2);
  B << 0, 0, //
      dt, 0, //
      0, 0,  //
      0, dt;

  // cost: . 5 xQx + .5 uRu
  Eigen::MatrixXd Q(4, 4);
  // Q << 1, 0, 0, 1;
  // Q << 0, 0, 0, 0;
  Q.setZero();
  Eigen::MatrixXd R(2, 2);
  R.setIdentity();
  // R << 1;

  Eigen::MatrixXd Qf(4, 4);
  Qf.setIdentity();
  Qf *= 10;
  // Q
  // Qf << 10, 0, 0, 10;

  ddp(A, B, Q, Qf, R, xs, us);
}

// TODO: try with a non-linear system (e.g. unicycle)
// Add bounds.

BOOST_AUTO_TEST_CASE(test_ddp) {
  // ddp_example_1();
  // ddp_example_2();
  ddp_example_3();
}
