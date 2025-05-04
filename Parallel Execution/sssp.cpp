#include "sssp.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <mpi.h>
#include <omp.h>
#include <iostream>
using namespace std;

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