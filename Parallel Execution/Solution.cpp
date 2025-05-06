#include "opencl_utils.h"
#include <climits>
#include <queue>
#include <metis.h>
#include <algorithm>
#include <mpi.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#define ll long long

using namespace std;


struct Edge
{
    int u, v;
    ll weight;
};
struct UpdateData
{
    int u, v;
    ll weight;
    bool is_removal;
};
vector<UpdateData> loadUpdates(const string &filename);
void saveResults(const string &filename, const vector<ll> &dist);
void printStats(const vector<ll> &dist);

class Graph
{
public:
    int V, E;
    vector<Edge> edges;
    vector<vector<pair<int, ll>>> adj;

    // Partitioning information
    vector<int> part;
    vector<int> local_vertices;
    vector<int> ghost_vertices;

    Graph() : V(0), E(0) {}
    void loadFromFile(const string &filename)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error opening file: " << filename << endl;
            return;
        }
    
        if (!(file >> V >> E))
        {
            cerr << "Error reading graph header" << endl;
            return;
        }
    
        if (V <= 0 || E <= 0)
        {
            cerr << "Invalid graph size: V=" << V << ", E=" << E << endl;
            V = 0;
            E = 0;
            return;
        }
    
        adj.clear();
        adj.resize(V);
        edges.clear();
        edges.reserve(E);
    
        int u, v;
        ll weight;
        int edge_count = 0;
        string line;
    
        getline(file, line);
    
        while (edge_count < E && getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
    
            istringstream iss(line);
    
            if (!(iss >> u >> v >> weight))
            {
                cerr << "Error parsing edge line: " << line << endl;
                continue;
            }
    
            if (u < 0 || u >= V || v < 0 || v >= V)
            {
                cerr << "Invalid vertex indices in edge: " << u << " " << v << endl;
                continue;
            }
    
            if (u == v)
            {
                cerr << "Warning: Self-loop found at vertex " << u << ", ignoring" << endl;
                continue;
            }
    
            if (weight < 0)
            {
                cerr << "Warning: Negative weight found in edge " << u << "-" << v
                          << ", Dijkstra's algorithm may not work correctly" << endl;
            }
    
            edges.push_back({u, v, weight});
            adj[u].emplace_back(v, weight);
            adj[v].emplace_back(u, weight);
            edge_count++;
        }
    
        if (edge_count < E)
        {
            cerr << "Warning: Expected " << E << " edges but found only " << edge_count << endl;
            E = edge_count;
        }
    
        while (getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
    
            istringstream iss(line);
    
            if (iss >> u >> v >> weight)
            {
                if (u < 0 || u >= V || v < 0 || v >= V)
                {
                    cerr << "Invalid vertex indices in additional edge: " << u << " " << v << endl;
                    continue;
                }
    
                if (u == v)
                {
                    cerr << "Warning: Self-loop found at vertex " << u << ", ignoring" << endl;
                    continue;
                }
    
                edges.push_back({u, v, weight});
                adj[u].emplace_back(v, weight);
                adj[v].emplace_back(u, weight);
                E++;
            }
        }
    
        cout << "Successfully loaded graph with " << V << " vertices and " << E << " edges" << endl;
    }
    void partitionGraph(int num_parts)
    {
        if (V == 0)
            return;
        if (num_parts > V)
        {
            cerr << "Warning: More partitions than vertices, setting num_parts = V" << endl;
            num_parts = V;
        }
    
        idx_t nvtxs = V;
        idx_t ncon = 1;
        idx_t *xadj = new idx_t[V + 1];
        idx_t *adjncy = new idx_t[2 * E];
        idx_t *vwgt = nullptr;
        idx_t *adjwgt = nullptr;
        idx_t objval;
        idx_t nparts = num_parts;
        part.resize(V);
    
        xadj[0] = 0;
        for (int i = 0; i < V; i++)
        {
            xadj[i + 1] = xadj[i] + adj[i].size();
            for (size_t j = 0; j < adj[i].size(); j++)
            {
                adjncy[xadj[i] + j] = adj[i][j].first;
            }
        }
    
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    
        
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, nullptr,
                                      adjwgt, &nparts, nullptr, nullptr, options,
                                      &objval, part.data());
    
        if (ret != METIS_OK)
        {
            cerr << "METIS partitioning failed with code " << ret << endl;
            
            // If partitioning still fails, provide more diagnostics
            if (ret == METIS_ERROR_INPUT)
                cerr << "Error in the graph's input format" << endl;
            else if (ret == METIS_ERROR_MEMORY)
                cerr << "METIS could not allocate required memory" << endl;
            else if (ret == METIS_ERROR)
                cerr << "General METIS error" << endl;
                
            cerr << "Using simple vertex partitioning instead" << endl;
            for (int v = 0; v < V; v++)
            {
                part[v] = v % num_parts;
            }
        }
        else
        {
            cout << "Successfully partitioned graph into " << num_parts << " parts" << endl;
            
            // Optionally, print partition statistics
            vector<int> part_sizes(num_parts, 0);
            for (int v = 0; v < V; v++)
            {
                if (part[v] >= 0 && part[v] < num_parts)
                    part_sizes[part[v]]++;
            }
            
            cout << "Partition sizes: ";
            for (int i = 0; i < num_parts; i++)
            {
                cout << part_sizes[i];
                if (i < num_parts - 1)
                    cout << ", ";
            }
            cout << endl;
        }
    
        delete[] xadj;
        delete[] adjncy;
    }
    void distributeGraph(MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    
        local_vertices.clear();
        ghost_vertices.clear();
    
        for (int v = 0; v < V; v++)
        {
            if (part[v] == rank)
            {
                local_vertices.push_back(v);
            }
        }
    
        for (int v : local_vertices)
        {
            for (const auto &neighbor : adj[v])
            {
                int u = neighbor.first;
                if (part[u] != rank &&
                    find(ghost_vertices.begin(), ghost_vertices.end(), u) == ghost_vertices.end())
                {
                    ghost_vertices.push_back(u);
                }
            }
        }
    }
    
    void addEdge(int u, int v, ll weight)
    {
        if (u < 0 || u >= V || v < 0 || v >= V)
        {
            cerr << "Invalid vertex indices in edge: " << u << " " << v << endl;
            return;
        }
    
        edges.push_back({u, v, weight});
        adj[u].emplace_back(v, weight);
        adj[v].emplace_back(u, weight);
        E++;
    }
    
    void applyUpdates(const vector<UpdateData> &updates)
    {
        for (const auto &edge : updates)
        {
            if (edge.is_removal) // Handle deletion
            {
                // Remove edge from adj[edge.u]
                adj[edge.u].erase(
                    remove_if(adj[edge.u].begin(), adj[edge.u].end(),
                                   [edge](const pair<int, ll> &neighbor)
                                   {
                                       return neighbor.first == edge.v;
                                   }),
                    adj[edge.u].end());
                // Remove edge from adj[edge.v]
                adj[edge.v].erase(
                    remove_if(adj[edge.v].begin(), adj[edge.v].end(),
                                   [edge](const pair<int, ll> &neighbor)
                                   {
                                       return neighbor.first == edge.u;
                                   }),
                    adj[edge.v].end());
                // Update edge count
                E--;
            }
            else // Handle insertion or update
            {
                bool found = false;
                for (auto &neighbor : adj[edge.u])
                {
                    if (neighbor.first == edge.v)
                    {
                        neighbor.second = edge.weight;
                        found = true;
                        break;
                    }
                }
                for (auto &neighbor : adj[edge.v])
                {
                    if (neighbor.first == edge.u)
                    {
                        neighbor.second = edge.weight;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    addEdge(edge.u, edge.v, edge.weight);
                }
            }
        }
    }
    
    void gatherSSSPResults(MPI_Comm comm, vector<ll> &global_dist)
    {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        vector<ll> local_dists = global_dist;
        vector<ll> all_dists(V);

        MPI_Allreduce(local_dists.data(), all_dists.data(), V, MPI_LONG_LONG, MPI_MIN, comm);

        global_dist = all_dists;

        if (rank == 0)
        {
            cout << "Gathered SSSP results from all processes" << endl;
        }
    }
};



class SSSP
{
public:
    vector<ll> dist;
    vector<int> parent;
    vector<bool> affected;
    vector<bool> affected_del;

    // OpenCL data structures
    bool opencl_available = false;
    OpenCLContext opencl_ctx;
    vector<pair<int, int>> edge_pairs;
    vector<ll> edge_weights;

    SSSP(int V): dist(V, LLONG_MAX),
            parent(V, -1),
            affected(V, false),
            affected_del(V, false) {}
    void initialize(int source);
    void updateStep1(const Graph &graph, const vector<UpdateData>&inserts,
                     const vector<UpdateData>&deletes, bool use_openmp);
    void updateStep2(Graph &graph, bool use_openmp, int async_level, bool use_opencl = false);
    void updateStep2CPU(Graph &graph, bool use_openmp, int async_level); // Added declaration
    bool hasConverged(MPI_Comm comm);
    void markAffectedSubtree(int root, Graph &graph);

    // New method to prepare graph data for OpenCL
    void prepareGraphForOpenCL(const Graph &graph);
};


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


void SSSP::initialize(int source)
{
    if (source < 0 || source >= static_cast<int>(dist.size()))
    {
        cerr << "Error: Invalid source vertex " << source << endl;
        return;
    }

    fill(dist.begin(), dist.end(), LLONG_MAX);
    fill(parent.begin(), parent.end(), -1);
    fill(affected.begin(), affected.end(), false);
    fill(affected_del.begin(), affected_del.end(), false);
    dist[source] = 0;

    // Mark source as affected to trigger initial computation
    affected[source] = true;
}

void SSSP::prepareGraphForOpenCL(const Graph &graph)
{
    // Convert adjacency list to edge pairs and weights for OpenCL
    edge_pairs.clear();
    edge_weights.clear();

    for (int u = 0; u < graph.V; u++)
    {
        for (const auto &neighbor : graph.adj[u])
        {
            int v = neighbor.first;
            ll weight = neighbor.second;

            // Avoid duplicate edges (since this is an undirected graph)
            if (u < v)
            {
                edge_pairs.push_back({u, v});
                edge_weights.push_back(weight);
            }
        }
    }

    // Initialize OpenCL if not already done
    if (!opencl_available)
    {
        opencl_available = setupOpenCL(opencl_ctx, "relax_edges.cl");
        if (!opencl_available)
        {
            cerr << "Warning: OpenCL initialization failed, falling back to CPU implementation" << endl;
        }
    }
}

void SSSP::updateStep1(const Graph &graph, const vector<UpdateData> &inserts,
                       const vector<UpdateData> &deletes, bool use_openmp)
{
#pragma omp parallel for if (use_openmp)
    for (size_t i = 0; i < deletes.size(); i++)
    {
        const UpdateData &e = deletes[i];
        if (e.u < 0 || e.u >= static_cast<int>(dist.size()) ||
            e.v < 0 || e.v >= static_cast<int>(dist.size()))
        {
#pragma omp critical
            {
                cerr << "Warning: Invalid edge in deletions: " << e.u << " " << e.v << endl;
            }
            continue;
        }
        // Always mark both vertices as affected for deletion
        affected_del[e.u] = true;
        affected_del[e.v] = true;
        affected[e.u] = true;
        affected[e.v] = true;
        // Reset distances if the edge was part of the shortest path
        if (parent[e.v] == e.u)
        {
            dist[e.v] = LLONG_MAX;
            parent[e.v] = -1;
        }
        if (parent[e.u] == e.v)
        {
            dist[e.u] = LLONG_MAX;
            parent[e.u] = -1;
        }
    }

#pragma omp parallel for if (use_openmp)
    for (size_t i = 0; i < inserts.size(); i++)
    {
        const UpdateData &e = inserts[i];
        int u = e.u, v = e.v;
        ll weight = e.weight;

        // Validate edge vertices
        if (u < 0 || u >= static_cast<int>(dist.size()) ||
            v < 0 || v >= static_cast<int>(dist.size()))
        {
#pragma omp critical
            {
                cerr << "Warning: Invalid edge in insertions: " << u << " " << v << endl;
            }
            continue;
        }

        if (dist[u] > dist[v])
            swap(u, v);

        if (dist[v] > dist[u] + weight)
        {
            dist[v] = dist[u] + weight;
            parent[v] = u;
            affected[v] = true;
        }
    }
}

void SSSP::updateStep2(Graph &graph, bool use_openmp, int async_level, bool use_opencl)
{
    if (use_opencl && opencl_available)
    {
        cout << "Running OpenCL SSSP on GPU..." << endl;
        prepareGraphForOpenCL(graph);
        runRelaxationKernel(opencl_ctx, dist, parent, edge_pairs, edge_weights);
    }
    else
    {
        cout << "Running CPU SSSP..." << endl;
        updateStep2CPU(graph, use_openmp, async_level);
    }
}

void SSSP::updateStep2CPU(Graph &graph, bool use_openmp, int async_level)
{
    bool changed;
    int iterations = 0;
    const int MAX_ITERATIONS = 100;

    // Preserve initial distances and reset only affected vertices
    vector<ll> temp_dist = dist;
    vector<int> temp_parent = parent;

    do
    {
        changed = false;
        iterations++;

        // Phase 1: Handle deletions and reset affected subtrees
        #pragma omp parallel for if (use_openmp) schedule(dynamic)
        for (int v = 0; v < graph.V; v++)
        {
            if (affected_del[v])
            {
                affected_del[v] = false;
                bool local_changed = false;
                for (const auto &neighbor : graph.adj[v])
                {
                    int c = neighbor.first;
                    if (c >= 0 && c < static_cast<int>(parent.size()) && parent[c] == v)
                    {
                        #pragma omp critical
                        {
                            dist[c] = LLONG_MAX;
                            parent[c] = -1;
                            affected_del[c] = true;
                            affected[c] = true;
                            local_changed = true;
                        }
                    }
                }
                if (local_changed)
                {
                    #pragma omp atomic write
                        changed = true;
                }
            }
        }

        // Phase 2: Recompute paths for all vertices
        vector<bool> visited(graph.V, false);
        priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;

        // Initialize with vertices that have finite distances
        for (int v = 0; v < graph.V; v++)
            if (dist[v] != LLONG_MAX)
                pq.push({dist[v], v});


        while (!pq.empty())
        {
            auto [d, u] = pq.top();
            pq.pop();
            if (visited[u])
                continue;
            visited[u] = true;

            for (const auto &neighbor : graph.adj[u])
            {
                int v = neighbor.first;
                ll weight = neighbor.second;
                if (v >= 0 && v < static_cast<int>(dist.size()))
                {
                    ll new_dist = dist[u] + weight;
                    if (new_dist < dist[v])
                    {
                        dist[v] = new_dist;
                        parent[v] = u;
                        pq.push({new_dist, v});
                        affected[v] = true;
                        changed = true;
                    }
                }
            }
        }

        if (iterations % 10 == 0)
        {
            cout << "Iteration " << iterations << ", changed = " << changed << endl;
        }

        if (async_level > 1 && iterations % async_level == 0)
        {
            vector<ll> global_dist(graph.V);
            for (int i = 0; i < graph.V; i++)
            {
                global_dist[i] = dist[i];
            }
            graph.gatherSSSPResults(MPI_COMM_WORLD, global_dist);

            for (int i = 0; i < graph.V; i++)
            {
                if (global_dist[i] < dist[i])
                {
                    dist[i] = global_dist[i];
                    affected[i] = true;
                    changed = true;
                }
            }
        }

    } while (changed && iterations < MAX_ITERATIONS);

    if (iterations >= MAX_ITERATIONS)
        cerr << "Warning: updateStep2 reached maximum iterations without converging" << endl;
    else
        cout << "SSSP converged after " << iterations << " iterations." << endl;

}

bool SSSP::hasConverged(MPI_Comm comm)
{
    int local_changed = 0;

    for (bool a : affected)
    {
        if (a)
        {
            local_changed = 1;
            break;
        }
    }

    int global_changed;
    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, comm);
    return global_changed == 0;
}

void SSSP::markAffectedSubtree(int root, Graph &graph)
{
    if (root < 0 || root >= static_cast<int>(dist.size()))
    {
        cerr << "Error: Invalid root vertex " << root << endl;
        return;
    }

    queue<int> q;
    q.push(root);
    affected_del[root] = true;
    affected[root] = true;

    while (!q.empty())
    {
        int v = q.front();
        q.pop();

        for (const auto &neighbor : graph.adj[v])
        {
            int c = neighbor.first;

            if (c < 0 || c >= static_cast<int>(parent.size()))
            {
                continue;
            }

            if (parent[c] == v)
            {
                dist[c] = LLONG_MAX;
                parent[c] = -1;
                affected_del[c] = true;
                affected[c] = true;
                q.push(c);
            }
        }
    }
}


// Implementations
vector<UpdateData> loadUpdates(const string &filename)
{
    vector<UpdateData> updates;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening updates file: " << filename << endl;
        return updates;
    }

    string line;
    while (getline(file, line))
    {
        // Skip empty lines
        if (line.empty()) continue;

        // Skip comment lines or lines that don't start with a digit
        if (line[0] == '#' || !isdigit(line[0])) continue;

        istringstream iss(line);
        UpdateData e;

        // Try to parse the line as "u v weight"
        if (iss >> e.u >> e.v)
        {
            string type, weight_str;
            if (iss >> type)
            {
                // Check if it's a removal (marked with '-')
                if (type == "del")
                    e.is_removal = true;
                else if (iss >> weight_str) 
                        // Try to parse as ll
                        e.weight = stoll(weight_str);

                updates.push_back(e);
            }
        }
        else
        {
            cerr << "Malformed update line: " << line << endl;
        }
    }

    cout << "Total updates loaded: " << updates.size() << endl;
    return updates;
}
void saveResults(const string &filename, const vector<ll> &dist)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening output file: " << filename << endl;
        return;
    }

    for (size_t i = 0; i < dist.size(); i++)
        file << i << " " << fixed << setprecision(2) << dist[i] << "\n";
}

void printStats(const vector<ll> &dist)
{
    int reachable = 0;
    ll max_dist = 0;
    ll sum_dist = 0;

    for (ll d : dist)
    {
        if (d < numeric_limits<ll>::infinity())
        {
            reachable++;
            if (d > max_dist)
                max_dist = d;
            sum_dist += d;
        }
    }

    cout << "Statistics:\n";
    cout << "Reachable vertices: " << reachable << "/" << dist.size() << "\n";
    cout << "Maximum distance: " << max_dist << "\n";
    cout << "Average distance: " << (reachable > 0 ? sum_dist / reachable : 0) << "\n";
}


