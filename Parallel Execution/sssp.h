
#ifndef SSSP_H
#define SSSP_H

#include "graph.h"
#include "opencl_utils.h"
#include <vector>
#include <climits>
#include <mpi.h>

using namespace std;

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

#endif // SSSP_H