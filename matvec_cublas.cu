// matvec_cublas.cu
// Minimal matrix-vector multiplication using CUDA cuBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Read CSV file and return a tuple of (std::vector A, std::vector x,
// std::vector y) Assumes A.csv is N x M matrix (rows x cols), x.csv is
// M-element vector Computes single-threaded y = A*x and returns it
std::tuple<std::vector<double>, std::vector<double>> read_csv_files(
    const std::string& matrix_file, const std::string& vector_file) {
    // Read matrix from CSV
    std::vector<double> matrix_data;
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
                matrix_data.push_back(std::stod(value));
            }
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

    return std::make_tuple(matrix_data, vector_data);
}

int main() {
    auto [A, x] = read_csv_files("A.csv", "x.csv");
    int m = x.size();
    int n = m;  // Assuming A is square for simplicity
    std::vector<double> y(m, 0);

    double *A_d, *x_d, *y_d;
    cudaMalloc(&A_d, m * n * sizeof(double));
    cudaMalloc(&x_d, n * sizeof(double));
    cudaMalloc(&y_d, m * sizeof(double));

    // cuBLAS expects column-major, so transpose A if needed
    cudaMemcpy(A_d, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // start timing

    const double alpha = 1.0, beta = 0.0;
    const auto nrepeat = 100;
    // y = alpha*A*x + beta*y
    for (int i = 0; i < nrepeat; i++) {
        cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, A_d, m, x_d, 1, &beta,
                    y_d, 1);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);       // stop timing
    cudaEventSynchronize(stop);  // wait for GPU

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel time: %f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Bandwidth calculation:
    //   Each matrix A row is read once.
    //   The x vector (of length M) is read N times.
    double Gbytes = 1.0e-9 * double(sizeof(double) * (m + m * n));

    // Print results (problem size, time and bandwidth in GB/s).
    printf(
        "M( %d ) N( %d ) nrepeat ( %d ) problem size ( %g MB ) time( "
        "%g s ) bandwidth( %g GB/s )\n",
        m, n, nrepeat, Gbytes * 1000, ms / 1000.0,
        Gbytes * nrepeat / (ms / 1000.0));

    cudaMemcpy(y.data(), y_d, m * sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_d);
    return 0;
}
