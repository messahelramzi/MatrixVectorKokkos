#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using ExecHostSpace = Kokkos::DefaultHostExecutionSpace;

// Single-threaded matrix-vector multiply: y = A * x
template <typename AViewType, typename xViewType, typename yViewType>
void matvec_serial(const AViewType& A, const xViewType& x, yViewType& y) {
    const int N = A.extent(0);  // rows
    const int M = A.extent(1);  // cols

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < M; ++j) {
            sum += A(i, j) * x(j);
        }
        y(i) = sum;
    }
}

// Kokkos kernels Blas2 Dense matrix-vector multiply: y = A * x
template <typename AViewType, typename xViewType, typename yViewType>
void matvec_kokkos_kernels(const AViewType& A, const xViewType& x,
                           yViewType& y) {
    // This region will be visible in Kokkos profiling tools
    // like nvprof or VTune
    Kokkos::Profiling::pushRegion("MatVecKernels");

    const double alpha = double(1.0);
    const double beta = double(0.0);

    KokkosBlas::gemv("N", alpha, A, x, beta, y);

    Kokkos::fence();
    Kokkos::Profiling::popRegion();
}

// Compute relative error ||y - y_ref|| / ||y_ref|| using Kokkos parallel_reduce
template <typename YViewType>
double compute_relative_error(const YViewType& y, const YViewType& y_ref) {
    const int N = y.extent(0);
    double sum_diff2 = 0.0;
    double sum_ref2 = 0.0;

    using exec_t = typename YViewType::device_type::execution_space;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<exec_t>(0, N),
        KOKKOS_LAMBDA(const int i, double& lsum) {
            double d = y(i) - y_ref(i);
            lsum += d * d;
        },
        sum_diff2);

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<exec_t>(0, N),
        KOKKOS_LAMBDA(const int i, double& lsum) {
            double v = y_ref(i);
            lsum += v * v;
        },
        sum_ref2);

    if (sum_ref2 == 0.0) {
        return (sum_diff2 == 0.0) ? 0.0
                                  : std::numeric_limits<double>::infinity();
    }
    return std::sqrt(sum_diff2 / sum_ref2);
}

// Read CSV file and return a tuple of (Kokkos::View A, Kokkos::View x,
// Kokkos::View y) Assumes A.csv is N x M matrix (rows x cols), x.csv is
// M-element vector Computes single-threaded y = A*x and returns it
template <typename Execspace, typename LayoutTag>
std::tuple<Kokkos::View<double**, LayoutTag, typename Execspace::memory_space>,
           Kokkos::View<double*, typename Execspace::memory_space>,
           Kokkos::View<double*, typename Execspace::memory_space>>
read_csv_files(const std::string& matrix_file, const std::string& vector_file) {
    using MemorySpace = typename Execspace::memory_space;
    Execspace execSpace{};

    // Read matrix from CSV
    std::vector<std::vector<double>> matrix_data;
    {
        std::ifstream file(matrix_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open matrix file: " + matrix_file);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            matrix_data.push_back(row);
        }
    }

    // Read vector from CSV
    std::vector<double> vector_data;
    {
        std::ifstream file(vector_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vector file: " + vector_file);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                vector_data.push_back(std::stod(value));
            }
        }
    }

    // Get dimensions
    const int N = matrix_data.size();                   // rows
    const int M = (N > 0) ? matrix_data[0].size() : 0;  // cols

    // Create Kokkos views
    auto A = Kokkos::View<double**, LayoutTag, MemorySpace>(
        Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "A"), N, M);
    auto x = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "x"), M);
    auto y = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "y"), N);

    // Copy data to host mirror, then to device
    auto A_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, A);
    auto x_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, x);
    auto y_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, y);

    Kokkos::parallel_for(
        "CopyMatrixToHost",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
        [=](int i) {
            for (int j = 0; j < M; ++j) {
                A_host(i, j) = matrix_data[i][j];
                x_host(j) = vector_data[j];
            }
        });

    // Compute single-threaded solution on host
    matvec_serial(A_host, x_host, y_host);

    // Copy to device
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(x, x_host);
    Kokkos::deep_copy(y, y_host);

    return std::make_tuple(A, x, y);
}
