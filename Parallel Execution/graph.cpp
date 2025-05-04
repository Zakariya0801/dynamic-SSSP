#include "graph.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <metis.h>
#include <algorithm>
#define ll long long
using namespace std;


void Graph::loadFromFile(const string &filename)
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

void Graph::partitionGraph(int num_parts)
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
void Graph::distributeGraph(MPI_Comm comm)
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

void Graph::addEdge(int u, int v, ll weight)
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

void Graph::applyUpdates(const vector<UpdateData> &updates)
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

void Graph::gatherSSSPResults(MPI_Comm comm, vector<ll> &global_dist)
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