#pragma once

#include "dynoRRT/dynorrt_macros.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include "nlohmann/json.hpp"

#include "eigen_conversions.hpp"
#include <fstream>

namespace dynorrt {

template <int DIM> struct BallObstacle {

  using Vector = Eigen::Matrix<double, DIM, 1>;
  using Cref = const Eigen::Ref<const Vector> &;

  BallObstacle(Cref &center, double radius) : center(center), radius(radius) {}
  BallObstacle() = default;

  Vector center;
  double radius;

  double distance(Cref &point) const {
    return (point - center).norm() - radius;
  }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BallObstacle<2>, center, radius);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BallObstacle<3>, center, radius);

template <int DIM> class CollisionManagerBallWorld {

public:
  using Vector = Eigen::Matrix<double, DIM, 1>;
  using Cref = const Eigen::Ref<const Vector> &;
  using Obstacle = BallObstacle<DIM>;

  double distance(Cref &point) const {
    double min_distance = std::numeric_limits<double>::max();
    for (const auto &obstacle : obstacles_) {
      min_distance =
          std::min(min_distance, obstacle.distance(point) - radius_robot);
    }
    return min_distance;
  }

  bool is_collision(Cref &point) const {
    for (const auto &obstacle : obstacles_) {
      if (obstacle.distance(point) - radius_robot < 0) {
        return true;
      }
    }
    return false;
  };

  void add_obstacle(const Obstacle &obstacle) {
    obstacles_.push_back(obstacle);
  }
  const std::vector<Obstacle> &get_obstacles() const { return obstacles_; }

  void set_obstacles(const std::vector<Obstacle> &obstacles) {
    obstacles_ = obstacles;
  }

  void set_radius_robot(double radius) { radius_robot = radius; }

  void load_world(std::string file) {

    std::ifstream f(file);
    if (!f.good()) {
      std::stringstream ss;
      ss << "File " << file << " does not exist" << std::endl;
      THROW_PRETTY_DYNORRT(ss.str());
    }

    json j;
    f >> j;

    obstacles_ = j["obstacles"];
    radius_robot = j["radius_robot"];
  }

protected:
  double radius_robot = 0.1;

  std::vector<Obstacle> obstacles_;
};

} // namespace dynorrt
