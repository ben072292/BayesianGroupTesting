#include "bgt/detail/model/lattice.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace model = bgt::model;

void run_model_manipulation_benchmark(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (argc != 5)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <parallelism_type> <subjs> <variants> <iterations>\n";
        }
        MPI_Finalize(); // Finalize MPI before exiting
        return;
    }

    int parallelism_type = std::atoi(argv[1]);
    int subjs = std::atoi(argv[2]);
    int variants = std::atoi(argv[3]);
    int iters = std::atoi(argv[4]);

    std::vector<double> pi0(subjs * variants);
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = 0.01;
    }

    if (parallelism_type == DIST_NON_DILUTION)
    {
        model::DistributedLattice::lattice_mpi_initialize(subjs, variants);
    }
    else if (parallelism_type == DIST_DILUTION)
    {
        model::DistributedLattice::lattice_mpi_initialize(subjs, variants);
    }
    else if (parallelism_type == REPL_NON_DILUTION)
    {
        model::Lattice::lattice_mpi_initialize();
    }
    else if (parallelism_type == REPL_DILUTION)
    {
        model::Lattice::lattice_mpi_initialize();
    }
    else
        exit(1);

    auto make_lattice = [&]() -> std::unique_ptr<model::Lattice>
    {
        if (parallelism_type == DIST_NON_DILUTION)
            return std::make_unique<model::DistributedNonDilutionLattice>(subjs, variants, pi0.data());
        if (parallelism_type == DIST_DILUTION)
            return std::make_unique<model::DistributedDilutionLattice>(subjs, variants, pi0.data());
        if (parallelism_type == REPL_NON_DILUTION)
            return std::make_unique<model::ReplicatedNonDilutionLattice>(subjs, variants, pi0.data());
        if (parallelism_type == REPL_DILUTION)
            return std::make_unique<model::ReplicatedDilutionLattice>(subjs, variants, pi0.data());
        throw std::logic_error("invalid lattice parallelism type");
    };

    std::unique_ptr<model::Lattice> p;
    auto start_lattice_construction = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        p = make_lattice();
    }

    auto end_lattice_construction = std::chrono::high_resolution_clock::now();

    auto start_updating = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++)
    {
        p->update_probs_in_place(1, 3, nullptr);
    }

    auto end_updating = std::chrono::high_resolution_clock::now();

    auto start_classification = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++)
    {
        p->get_atom_prob_mass(1);
    }

    auto end_classification = std::chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        std::stringstream file_name;
        file_name << "Model-Manipulation-Benchmark-" << p->type()
                  << "-N=" << subjs
                  << "-k=" << variants
                  << "-Processes=" << world_size
#ifdef ENABLE_OMP
                  << "-Threads=" << std::stoi(getenv("OMP_NUM_THREADS"))
#endif
                  << "-" << get_curr_time()
                  << ".csv";
        freopen(file_name.str().c_str(), "w", stdout);
        std::cout << hardware_config_summary() << std::endl;
        std::cout << "N," << subjs
                  << ",k," << variants
                  << ",Model Parallelism:," << ((parallelism_type == DIST_NON_DILUTION || parallelism_type == DIST_DILUTION) ? "distributed" : "replicated")
                  << ",Use Dilution Effect:," << ((parallelism_type == DIST_NON_DILUTION || parallelism_type == REPL_NON_DILUTION) ? "no dilution" : "dilution")
#ifdef ENABLE_OMP
                  << ",OpenMP," << "Enabled"
#endif
                  << ",Iterations," << iters
                  << std::endl;
        std::cout << "Model Construction Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_lattice_construction - start_lattice_construction).count() / 1e9 / iters << "s\n";
        std::cout << "Model Update Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_updating - start_updating).count() / 1e9 / iters << "s\n";
        std::cout << "Model Classification Identification Time," << std::chrono::duration_cast<std::chrono::nanoseconds>(end_classification - start_classification).count() / 1e9 / iters << "s\n";
    }

    // Free lattice model MPI env
    p.reset();
    model::Lattice::lattice_mpi_finalize();

    // Finalize the MPI environment.
    MPI_Finalize();
}
