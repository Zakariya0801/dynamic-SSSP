#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <limits>
#include "graph.h"
#include "sssp.h"
#include "utils.h"
#define ll long long
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4)
    {
        if (rank == 0)
        {
            cerr << "Usage: " << argv[0]
                      << " <graph_file> <updates_file> <source_vertex> [output_file] [--openmp] [--async=<level>] [--opencl]" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string graph_file = argv[1];
    string updates_file = argv[2];

    // Check if the source vertex is a valid number
    int source;
    try
    {
        source = stoi(argv[3]);
    }
    catch (const exception &e)
    {
        if (rank == 0)
        {
            cerr << "Error: Source vertex must be a valid integer, got '" << argv[3] << "'" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Default values
    string output_file = "";
    bool use_openmp = false;
    bool use_opencl = false;
    int async_level = 1;

    // Process optional arguments
    for (int i = 4; i < argc; i++)
    {
        string arg = argv[i];

        if (arg == "--openmp")
        {
            use_openmp = true;
        }
        else if (arg == "--opencl")
        {
            use_opencl = true;
        }
        else if (arg.compare(0, 8, "--async=") == 0)
        {
            try
            {
                async_level = stoi(arg.substr(8));
                if (async_level <= 0)
                {
                    if (rank == 0)
                    {
                        cerr << "Warning: Invalid async level " << async_level << ", setting to 1" << endl;
                    }
                    async_level = 1;
                }
            }
            catch (const exception &e)
            {
                if (rank == 0)
                {
                    cerr << "Warning: Invalid async value, using default level 1" << endl;
                }
                async_level = 1;
            }
        }
        else if (arg.compare(0, 2, "--") != 0)
        {
            output_file = arg;
        }
        else
        {
            if (rank == 0)
            {
                cerr << "Warning: Unknown option '" << arg << "'" << endl;
            }
        }
    }

    if (rank == 0)
    {
        cout << "Configuration:" << endl;
        cout << "  Graph file: " << graph_file << endl;
        cout << "  Updates file: " << updates_file << endl;
        cout << "  Source vertex: " << source << endl;
        cout << "  Output file: " << (output_file.empty() ? "none" : output_file) << endl;
        cout << "  OpenMP: " << (use_openmp ? "enabled" : "disabled") << endl;
        cout << "  OpenCL: " << (use_opencl ? "enabled" : "disabled") << endl;
        cout << "  Async level: " << async_level << endl;
    }

    // Load graph and partition it
    Graph graph;
    if (rank == 0)
    {
        cout << "Loading graph from " << graph_file << endl;
        graph.loadFromFile(graph_file);
        cout << "Graph loaded: " << graph.V << " vertices, " << graph.E << " edges" << endl;
        graph.partitionGraph(size);
    }

    // First broadcast graph size to all processes
    int graph_info[2];
    if (rank == 0)
    {
        graph_info[0] = graph.V;
        graph_info[1] = graph.E;
    }
    MPI_Bcast(graph_info, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Now non-root processes know the graph size
    if (rank != 0)
    {
        graph.V = graph_info[0];
        graph.E = graph_info[1];
        graph.adj.resize(graph.V);
        graph.part.resize(graph.V);
    }

    // Broadcast partition information
    MPI_Bcast(graph.part.data(), graph.V, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute graph data
    vector<Edge> all_edges;
    if (rank == 0)
    {
        all_edges = graph.edges;
    }

    int num_edges = (rank == 0) ? graph.E : 0;
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        all_edges.resize(num_edges);
    }

    // Create custom MPI datatype for Edge
    MPI_Datatype MPI_EDGE;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_LONG_LONG};

    Edge temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.u, &offsets[0]);
    MPI_Get_address(&temp.v, &offsets[1]);
    MPI_Get_address(&temp.weight, &offsets[2]);
    offsets[0] = MPI_Aint_diff(offsets[0], base_address);
    offsets[1] = MPI_Aint_diff(offsets[1], base_address);
    offsets[2] = MPI_Aint_diff(offsets[2], base_address);

    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);

    // Broadcast edges
    MPI_Bcast(all_edges.data(), num_edges, MPI_EDGE, 0, MPI_COMM_WORLD);

    // Reconstruct graph if not root
    if (rank != 0)
    {
        for (const auto &edge : all_edges)
        {
            graph.adj[edge.u].emplace_back(edge.v, edge.weight);
            graph.adj[edge.v].emplace_back(edge.u, edge.weight);
        }
        graph.edges = all_edges;
    }

    graph.distributeGraph(MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Graph distributed. Process 0 has " << graph.local_vertices.size()
                  << " local vertices and " << graph.ghost_vertices.size() << " ghost vertices" << endl;
    }

    // Initialize SSSP
    SSSP sssp(graph.V);
    sssp.initialize(source);

    if (rank == 0)
    {
        cout << "Running initial SSSP calculation from source " << source << endl;
    }

    sssp.updateStep2(graph, use_openmp, async_level, use_opencl);

    MPI_Barrier(MPI_COMM_WORLD);

    vector<ll> initial_dist(graph.V);
    for (int i = 0; i < graph.V; i++)
    {
        initial_dist[i] = sssp.dist[i];
    }

    graph.gatherSSSPResults(MPI_COMM_WORLD, initial_dist);

    if (rank == 0)
    {
        cout << "Initial SSSP completed. Statistics:" << endl;
        printStats(initial_dist);
    }

    // Load updates
    vector<UpdateData> all_updates;
    if (rank == 0)
    {
        cout << "Loading updates from " << updates_file << endl;
        all_updates = loadUpdates(updates_file);
        cout << "Loaded " << all_updates.size() << " updates" << endl;
    }

    int num_updates = (rank == 0) ? all_updates.size() : 0;
    MPI_Bcast(&num_updates, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        all_updates.resize(num_updates);
    }

    MPI_Bcast(all_updates.data(), num_updates, MPI_EDGE, 0, MPI_COMM_WORLD);
    cout << "hererere";
    vector<UpdateData> inserts, deletes;
    for (const auto &e : all_updates)
    {
        if (!e.is_removal)
        {
            inserts.push_back(e);
        }
        else
        {
            UpdateData delete_edge = {e.u, e.v, 0, true};
            for (const auto &adj_edge : graph.adj[e.u])
            {
                if (adj_edge.first == e.v)
                {
                    delete_edge.weight = adj_edge.second;
                    break;
                }
            }
            deletes.push_back(delete_edge);
        }
    }

    if (rank == 0)
    {
        cout << "Processing " << inserts.size() << " insertions and "
                  << deletes.size() << " deletions" << endl;
    }

    double start_time = MPI_Wtime();

    graph.applyUpdates(all_updates); // Apply both insertions and deletions

    // Redistribute updated graph
    num_edges = graph.E;
    vector<Edge> updated_edges;
    if (rank == 0)
    {
        updated_edges.clear();
        for (int u = 0; u < graph.V; u++)
        {
            for (const auto &neighbor : graph.adj[u])
            {
                int v = neighbor.first;
                ll weight = neighbor.second;
                if (u < v) // Avoid duplicates in undirected graph
                {
                    updated_edges.push_back({u, v, weight});
                }
            }
        }
        num_edges = updated_edges.size();
    }
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        updated_edges.resize(num_edges);
    }
    MPI_Bcast(updated_edges.data(), num_edges, MPI_EDGE, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        graph.adj.clear();
        graph.adj.resize(graph.V);
        for (const auto &edge : updated_edges)
        {
            graph.adj[edge.u].emplace_back(edge.v, edge.weight);
            graph.adj[edge.v].emplace_back(edge.u, edge.weight);
        }
        graph.edges = updated_edges;
        graph.E = num_edges;
    }
    graph.distributeGraph(MPI_COMM_WORLD);

    sssp.updateStep1(graph, inserts, deletes, use_openmp);

    MPI_Barrier(MPI_COMM_WORLD);

    sssp.updateStep2(graph, use_openmp, async_level, use_opencl);

    vector<ll> global_dist(graph.V, numeric_limits<ll>::infinity());
    for (int i = 0; i < graph.V; i++)
    {
        global_dist[i] = sssp.dist[i];
    }
    graph.gatherSSSPResults(MPI_COMM_WORLD, global_dist);

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        cout << "SSSP update completed in " << (end_time - start_time) << " seconds\n";
        printStats(global_dist);

        if (!output_file.empty())
        {
            saveResults(output_file, global_dist);
            cout << "Results saved to " << output_file << "\n";
        }
    }

    MPI_Type_free(&MPI_EDGE);
    MPI_Finalize();
    return 0;
}