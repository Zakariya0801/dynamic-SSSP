#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <mpi.h>
using namespace std;
#define ll long long

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
    void loadFromFile(const string &filename);
    void partitionGraph(int num_parts);
    void distributeGraph(MPI_Comm comm);
    void addEdge(int u, int v, ll weight);
    void applyUpdates(const vector<UpdateData> &updates);
    void gatherSSSPResults(MPI_Comm comm, vector<ll> &global_dist);
};

#endif // GRAPH_H