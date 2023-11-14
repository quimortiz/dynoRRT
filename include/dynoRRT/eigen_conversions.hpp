// NOTE: code has been copied from https://gitlab.com/Simox/simox
#pragma once

#include "nlohmann/json.hpp"
#include <Eigen/Core>

using json = nlohmann::json;

/**
 * Provide `to_json()` and `from_json()` overloads for `json`,
 * which allows simple syntax like:
 *
 * @code
 * Eigen::Matrix3f in, out;
 *
 * simox::json::json j;
 * j = in;
 * out = j.get<Eigen::Matrix3f>();
 * @endcode
 *
 * @test VirtualRobotJsonEigenConversionTest
 *
 * @see https://github.com/nlohmann/json#arbitrary-types-conversions
 */
namespace Eigen {

// Eigen::MatrixBase (non-specialized).

/// Writes the matrix as list of rows.
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void to_json(json &j, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                          _MaxRows, _MaxCols> &matrix);

template <typename Derived>
void to_json(json &j, const Eigen::MatrixBase<Derived> &matrix);

/// Reads the matrix from list of rows.
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void from_json(
    const json &j,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &matrix);

template <typename Derived>
void from_json(const json &j, Eigen::MatrixBase<Derived> &matrix);

// Specialization for Eigen::Vector3f (implemented in .cpp)

/// If `j` is an object, reads vector from `x, y, z` keys. Otherwise, reads it
/// as matrix.
void from_json(const json &j, Eigen::Vector3f &vector);

// Specialization for Eigen::Matrix4f as transformation matrix (implemented in
// .cpp).

/**
 * @brief Reads a 4x4 matrix from list of rows or `pos` and `ori` keys.
 *
 * If `j` is an object, reads `matrix` as transformation matrix from `pos` and
 * `ori` keys. Otherweise, reads it from list of rows.
 */
void from_json(const json &j, Eigen::Matrix4f &matrix);

// Eigen::Quaternion

/// Writes the quaternion with `qw, qx, qy, qz` keys.
template <typename Derived>
void to_json(json &j, const Eigen::QuaternionBase<Derived> &quat);

/// Reads the quaternion from `qw, qx, qy, qz` keys.
template <typename Derived>
void from_json(const json &j, Eigen::QuaternionBase<Derived> &quat);

// Eigen::Transform (Isometry, Affine, ...)

template <typename T, int N, int Type>
void to_json(json &j, const Eigen::Transform<T, N, Type> &transform) {
  to_json(j, transform.matrix());
}

template <typename T, int N, int Type>
void from_json(const json &j, Eigen::Transform<T, N, Type> &transform) {
  from_json(j, transform.matrix());
}

// void inline from_json(const json &j, Eigen::VectorXd &vector) {
//   vector.resize(j.size());
//   for (std::size_t row = 0; row < j.size(); ++row) {
//     const auto &jrow = j.at(row);
//     vector(row) = jrow.get<double>();
//   }
// }
// IMPLEMENTATION

namespace jsonbase {
// "private" excplititly non-specialized implementation
// (to make it callable from specialized implementations)
// (anonymous namespace does not work because these functions are required
// by specializations in the .cpp)

/// Writes the matrix as list of rows.
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void to_json(json &j, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                          _MaxRows, _MaxCols> &matrix) {
  for (int row = 0; row < matrix.rows(); ++row) {
    if (matrix.cols() > 1) {
      json jrow = json::array();
      for (int col = 0; col < matrix.cols(); ++col) {
        jrow.push_back(matrix(row, col));
      }
      j.push_back(jrow);
    } else {
      j.push_back(matrix(row, 0));
    }
  }
}

template <typename Derived>
void to_json(json &j, const Eigen::MatrixBase<Derived> &matrix) {
  for (int row = 0; row < matrix.rows(); ++row) {
    if (matrix.cols() > 1) {
      json jrow = json::array();
      for (int col = 0; col < matrix.cols(); ++col) {
        jrow.push_back(matrix(row, col));
      }
      j.push_back(jrow);
    } else {
      j.push_back(matrix(row, 0));
    }
  }
}

/// Reads the matrix from list of rows.
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void from_json(const json &j, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                            _MaxRows, _MaxCols> &matrix) {
  using Index = typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                       _MaxRows, _MaxCols>::Index;

  for (std::size_t row = 0; row < j.size(); ++row) {
    const auto &jrow = j.at(row);
    if (jrow.is_array()) {
      for (std::size_t col = 0; col < jrow.size(); ++col) {
        const auto &value = jrow.at(col);
        matrix(static_cast<Index>(row), static_cast<Index>(col)) =
            value.get<_Scalar>();
      }
    } else {
      matrix(static_cast<Index>(row), 0) = jrow.get<_Scalar>();
    }
  }
}

template <typename Derived>
void from_json(const json &j, Eigen::MatrixBase<Derived> &matrix) {
  using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
  using Index = typename Eigen::MatrixBase<Derived>::Index;

  for (std::size_t row = 0; row < j.size(); ++row) {
    const auto &jrow = j.at(row);
    if (jrow.is_array()) {
      for (std::size_t col = 0; col < jrow.size(); ++col) {
        const auto &value = jrow.at(col);
        matrix(static_cast<Index>(row), static_cast<Index>(col)) =
            value.get<Scalar>();
      }
    } else {
      matrix(static_cast<Index>(row), 0) = jrow.get<Scalar>();
    }
  }
}
} // namespace jsonbase

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void to_json(json &j, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                          _MaxRows, _MaxCols> &matrix) {
  jsonbase::to_json(j, matrix);
}

template <typename Derived>
void to_json(json &j, const Eigen::MatrixBase<Derived> &matrix) {
  jsonbase::to_json(j, matrix);
}

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
void from_json(const json &j, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                            _MaxRows, _MaxCols> &matrix) {
  jsonbase::from_json(j, matrix);
}

template <typename Derived>
void from_json(const json &j, Eigen::MatrixBase<Derived> &matrix) {
  jsonbase::from_json(j, matrix);
}

template <typename Derived>
void to_json(json &j, const Eigen::QuaternionBase<Derived> &quat) {
  j["qw"] = quat.w();
  j["qx"] = quat.x();
  j["qy"] = quat.y();
  j["qz"] = quat.z();
}

template <typename Derived>
void from_json(const json &j, Eigen::QuaternionBase<Derived> &quat) {
  using Scalar = typename Eigen::QuaternionBase<Derived>::Scalar;
  quat.w() = j.at("qw").get<Scalar>();
  quat.x() = j.at("qx").get<Scalar>();
  quat.y() = j.at("qy").get<Scalar>();
  quat.z() = j.at("qz").get<Scalar>();
}

} // namespace Eigen
