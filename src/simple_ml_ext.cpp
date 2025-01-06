#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename _T> class Matrix {
public:
  Matrix(int row, int col) : r(row), c(col) {
    m = std::make_unique<_T[]>(r * c);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        m[i * c + j] = 0;
      }
    }
  }

  Matrix(const _T *data, size_t row, size_t col) : r(row), c(col) {
    m = std::make_unique<_T[]>(r * c);
    std::copy(data, data + r * c, &m[0]);
  }

  Matrix(const Matrix &rhs) {
    r = rhs.r;
    c = rhs.c;
    m = std::make_unique<_T[]>(r * c);
    std::copy(&rhs.m[0], &rhs.m[0] + r * c, &m[0]);
  }

  void display() const {
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        std::cerr << at(i, j) << "\t";
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl << std::endl;
  }

  Matrix &operator=(const Matrix &rhs) {
    *this = Matrix(rhs);
    return *this;
  }

  Matrix(Matrix &&rhs) {
    r = rhs.r;
    c = rhs.c;
    m = std::move(rhs.m);
  }

  Matrix &operator=(Matrix &&rhs) {
    r = rhs.r;
    c = rhs.c;
    m = std::move(rhs.m);
    return *this;
  }

  Matrix T() const {
    int rr = c, cc = r;
    Matrix ret = Matrix(rr, cc);
    for (int i = 0; i < rr; ++i) {
      for (int j = 0; j < cc; ++j) {
        ret.at(i, j) = at(j, i);
      }
    }
    return ret;
  }

  Matrix matmul(const Matrix &rhs) const {
    assert(c == rhs.r);
    auto ret = Matrix(r, rhs.c);
    for (int i = 0; i < ret.r; ++i) {
      for (int j = 0; j < ret.c; ++j) {
        ret.at(i, j) = 0.;
        for (int k = 0; k < c; ++k) {
          ret.at(i, j) += at(i, k) * rhs.at(k, j);
        }
      }
    }
    return ret;
  }

  Matrix operator-(double v) const {
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) -= v;
      }
    }
    return ret;
  }

  Matrix operator-(const Matrix &rhs) const {
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) -= rhs.at(i, j);
      }
    }
    return ret;
  }

  Matrix operator*(double v) const {
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) *= v;
      }
    }
    return ret;
  }

  Matrix operator/(double v) const {
    assert(v != 0);
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) /= v;
      }
    }
    return ret;
  }

  Matrix exp() const {
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) = std::exp(ret.at(i, j));
      }
    }
    return ret;
  }

  Matrix normlize() const {
    auto ret = Matrix(*this);
    for (int i = 0; i < r; ++i) {
      _T sum = 0;
      for (int j = 0; j < c; ++j) {
        sum += ret.at(i, j);
      }
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) = ret.at(i, j) / sum;
      }
    }
    return ret;
  }

  static Matrix<_T> zeros(size_t r, size_t c) {
    auto ret = Matrix(r, c);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        ret.at(i, j) = 0;
      }
    }
    return ret;
  }

  template <typename R>
  static Matrix<_T> I(size_t num_class, const std::vector<R> &one_shot) {
    auto ret = Matrix<_T>(one_shot.size(), num_class);
    for (int i = 0; i < one_shot.size(); ++i) {
      ret.at(i, one_shot[i]) = 1;
    }
    return ret;
  }

  _T &at(size_t i, size_t j) { return m[i * c + j]; }
  const _T &at(size_t i, size_t j) const { return m[i * c + j]; }

private:
  std::unique_ptr<_T[]> m;
  int r;
  int c;
};

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */
  auto th = Matrix<float>(theta, n, k);
  for (int i = 0; i < m / batch; ++i) {
    size_t offset = i * batch;
    auto x_batch = Matrix<float>(X + offset * n, batch, n);
    // x_batch.display();
    // auto y_batch = std::vector<unsigned char>(y + offset, batch);
    auto y_batch = std::vector<int>();
    for (int j = 0; j < batch; ++j) {
      y_batch.push_back(y[offset + j]);
    }
    auto z = x_batch.matmul(th).exp().normlize();
    // x_batch.matmul(th).display();
    // x_batch.matmul(th).exp().display();
    // x_batch.matmul(th).exp().normlize().display();
    th = th - x_batch.T().matmul(z - Matrix<float>::I(k, y_batch)) / batch * lr;
    // x_batch.T().display();
    // Matrix<float>::I(k, y_batch).display();
    // (z - Matrix<float>::I(k, y_batch)).display();
    // x_batch.T().matmul(z - Matrix<float>::I(k, y_batch)).display();
    // (x_batch.T().matmul(z - Matrix<float>::I(k, y_batch)) / batch).display();
    // th.display();
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      theta[i * k + j] = th.at(i, j);
    }
  }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role
 * is wrap the function above in a Python module, and you do not need to
 * make any edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
