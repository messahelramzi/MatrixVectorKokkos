#include <CLI/CLI.hpp>

#include "matvec_kk.hxx"

int main(int argc, char* argv[]) {
    // Parse command line arguments with CLI11
    CLI::App app{"matvec - dense matrix-vector multiply"};
    std::string execution_space = "device";
    std::string matrix_file = "A.csv";
    std::string vector_file = "x.csv";
    int nrepeat = 1;
    std::string view_layout = "none";
    std::string view_layout_in = "none";

    app.add_option("--exec_space", execution_space,
                   "Execution space (device|host)")
        ->default_val("device");
    app.add_option("--matrix", matrix_file, "Path to matrix CSV file")
        ->default_val("A.csv");
    app.add_option("--vector", vector_file, "Path to vector CSV file")
        ->default_val("x.csv");
    app.add_option("--nrepeat", nrepeat, "Number of repetitions")
        ->default_val("100");
    app.add_option("--view_layout", view_layout_in, "Default view layout")
        ->default_val("none");

    CLI11_PARSE(app, argc, argv);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        using default_view_t = Kokkos::View<double**>;
        using default_layout_t = typename default_view_t::array_layout;
        if constexpr (std::is_same_v<default_layout_t, Kokkos::LayoutLeft>)
            view_layout = "LayoutLeft";
        else if constexpr (std::is_same_v<default_layout_t,
                                          Kokkos::LayoutRight>)
            view_layout = "LayoutRight";
        else
            std::cout << "Default layout: other\n";

        if (view_layout_in != "none") {
            if (view_layout_in == "LayoutLeft" ||
                view_layout_in == "LayoutRight") {
                view_layout = view_layout_in;
            } else {
                std::cerr << "Invalid view layout specified: " << view_layout_in
                          << ". Using default layout: " << view_layout
                          << std::endl;
            }
        }

        std::cout << "=====================================" << std::endl;
        std::cout << "Execution space: " << execution_space << std::endl;
        std::cout << "Matrix file: " << matrix_file << std::endl;
        std::cout << "Vector file: " << vector_file << std::endl;
        std::cout << "Repetitions: " << nrepeat << std::endl;
        std::cout << "Default view layout: " << view_layout << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << std::endl;

        double elapsed = std::numeric_limits<double>::max();
        double rel = std::numeric_limits<double>::max();
        int M = -1, N = -1;
        if (execution_space == "device") {
            if (view_layout == "LayoutLeft") {
                auto [A, x, y_ref] =
                    read_csv_files<ExecSpace, Kokkos::LayoutLeft>("A.csv",
                                                                  "x.csv");
                auto y_sol = Kokkos::View<double*, ExecSpace::memory_space>(
                    Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                                       "y_sol"),
                    y_ref.extent(0));
                Kokkos::Timer timer;
                for (int i = 0; i < nrepeat; ++i) {
                    matvec_kokkos_kernels(A, x, y_sol);
                    Kokkos::fence();
                }
                elapsed = timer.seconds();
                rel = compute_relative_error(y_sol, y_ref);
                M = A.extent(0);
                N = A.extent(1);
            } else {
                auto [A, x, y_ref] =
                    read_csv_files<ExecSpace, Kokkos::LayoutRight>("A.csv",
                                                                   "x.csv");
                auto y_sol = Kokkos::View<double*, ExecSpace::memory_space>(
                    Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing,
                                       "y_sol"),
                    y_ref.extent(0));
                Kokkos::Timer timer;
                for (int i = 0; i < nrepeat; ++i) {
                    matvec_kokkos_kernels(A, x, y_sol);
                    Kokkos::fence();
                }
                elapsed = timer.seconds();
                rel = compute_relative_error(y_sol, y_ref);
                M = A.extent(0);
                N = A.extent(1);
            }
        } else {
            if (view_layout == "LayoutLeft") {
                auto [A, x, y_ref] =
                    read_csv_files<ExecHostSpace, Kokkos::LayoutLeft>("A.csv",
                                                                      "x.csv");
                auto y_sol = Kokkos::View<double*, ExecHostSpace::memory_space>(
                    Kokkos::view_alloc(ExecHostSpace{},
                                       Kokkos::WithoutInitializing, "y_sol"),
                    y_ref.extent(0));
                Kokkos::Timer timer;
                for (int i = 0; i < nrepeat; ++i) {
                    matvec_kokkos_kernels(A, x, y_sol);
                    Kokkos::fence();
                }
                elapsed = timer.seconds();
                rel = compute_relative_error(y_sol, y_ref);
                M = A.extent(0);
                N = A.extent(1);
            } else {
                auto [A, x, y_ref] =
                    read_csv_files<ExecHostSpace, Kokkos::LayoutRight>("A.csv",
                                                                       "x.csv");
                auto y_sol = Kokkos::View<double*, ExecHostSpace::memory_space>(
                    Kokkos::view_alloc(ExecHostSpace{},
                                       Kokkos::WithoutInitializing, "y_sol"),
                    y_ref.extent(0));
                Kokkos::Timer timer;
                for (int i = 0; i < nrepeat; ++i) {
                    matvec_kokkos_kernels(A, x, y_sol);
                    Kokkos::fence();
                }
                elapsed = timer.seconds();
                rel = compute_relative_error(y_sol, y_ref);
                M = A.extent(0);
                N = A.extent(1);
            }
        }

        // Bandwidth calculation:
        //   Each matrix A row is read once.
        //   The x vector (of length M) is read N times.
        double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N));

        // Print results (problem size, time and bandwidth in GB/s).
        printf(
            "M( %d ) N( %d ) nrepeat ( %d ) problem size ( %g MB ) time( "
            "%g s ) bandwidth( %g GB/s )\n",
            M, N, nrepeat, Gbytes * 1000, elapsed, Gbytes * nrepeat / elapsed);
        std::cout << "Relative error: " << std::scientific
                  << std::setprecision(8) << rel << std::defaultfloat
                  << std::endl;
        if (rel < 1e-12)
            std::cout << "Validation PASSED\n";
        else
            std::cout << "Validation FAILED\n";

    }  // End of Kokkos scope
    Kokkos::finalize();

    return 0;
}
