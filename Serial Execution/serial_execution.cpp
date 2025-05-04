#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <climits>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <tuple>

#define ll long long
using namespace std;

// Edge structure
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

// Graph class
class Graph
{
public:
    int V, E;
    vector<Edge> edges;
    vector<vector<pair<int, ll>>> adj;

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

        getline(file, line); // Skip the first line after reading V and E

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
            adj[v].emplace_back(u, weight); // For undirected graph
            edge_count++;
        }

        if (edge_count < E)
        {
            cerr << "Warning: Expected " << E << " edges but found only " << edge_count << endl;
            E = edge_count;
        }

        cout << "Successfully loaded graph with " << V << " vertices and " << E << " edges" << endl;
    }

    void addEdge(int u, int v, ll weight)
    {
        if (u < 0 || u >= V || v < 0 || v >= V)
        {
            cerr << "Invalid vertex indices in edge: " << u << " " << v << endl;
            return;
        }

        // Check if edge already exists in adj[u]
        for (auto &neighbor : adj[u])
        {
            if (neighbor.first == v)
            {
                neighbor.second = weight; // Update weight
                // Also update the reverse direction
                for (auto &rev_neighbor : adj[v])
                {
                    if (rev_neighbor.first == u)
                    {
                        rev_neighbor.second = weight;
                        return;
                    }
                }
                return;
            }
        }

        // Check if edge exists in adj[v] (reverse direction)
        for (auto &neighbor : adj[v])
        {
            if (neighbor.first == u)
            {
                neighbor.second = weight;
                // Add the forward direction since it wasn't found
                adj[u].emplace_back(v, weight);
                edges.push_back({u, v, weight});
                E++;
                return;
            }
        }

        // If edge doesn't exist in either direction, add it
        edges.push_back({u, v, weight});
        adj[u].emplace_back(v, weight);
        adj[v].emplace_back(u, weight); // For undirected graph
        E++;
    }

    void applyUpdates(const vector<UpdateData> &updates)
    {
        for (const auto &edge : updates)
        {
            if (edge.is_removal)
            {
                // Handle edge deletion
                removeEdge(edge.u, edge.v);
                continue;
            }

            // Treat as insertion or update
            addEdge(edge.u, edge.v, edge.weight);
        }
    }

    void removeEdge(int u, int v)
    {
        if (u < 0 || u >= V || v < 0 || v >= V)
        {
            cerr << "Invalid vertex indices in edge deletion: " << u << " " << v << endl;
            return;
        }

        // Remove from edge list
        edges.erase(remove_if(edges.begin(), edges.end(),
                                   [u, v](const Edge &e)
                                   {
                                       return (e.u == u && e.v == v) || (e.u == v && e.v == u);
                                   }),
                    edges.end());

        // Remove from u's adjacency list
        adj[u].erase(remove_if(adj[u].begin(), adj[u].end(),
                                    [v](const pair<int, ll> &p)
                                    {
                                        return p.first == v;
                                    }),
                     adj[u].end());

        // Remove from v's adjacency list
        adj[v].erase(remove_if(adj[v].begin(), adj[v].end(),
                                    [u](const pair<int, ll> &p)
                                    {
                                        return p.first == u;
                                    }),
                     adj[v].end());

        E--;
    }
};

// SSSP (Single Source Shortest Path) class
class SSSP
{
public:
    vector<ll> dist;
    vector<int> parent;

    SSSP(int V) : dist(V, LLONG_MAX),
                  parent(V, -1) {}

    void initialize(int source)
    {
        if (source < 0 || source >= static_cast<int>(dist.size()))
        {
            cerr << "Error: Invalid source vertex " << source << endl;
            return;
        }

        fill(dist.begin(), dist.end(), LLONG_MAX);
        fill(parent.begin(), parent.end(), -1);
        dist[source] = 0;
    }

    void dijkstra(const Graph &graph, int source)
    {
        initialize(source);

        using P = pair<ll, int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        pq.emplace(0, source);

        while (!pq.empty())
        {
            ll d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (d > dist[u])
                continue;

            for (const auto &neighbor : graph.adj[u])
            {
                int v = neighbor.first;
                ll weight = neighbor.second;

                if (dist[v] > dist[u] + weight)
                {
                    dist[v] = dist[u] + weight;
                    parent[v] = u;
                    pq.emplace(dist[v], v);
                }
            }
        }

        cout << "Dijkstra's algorithm completed." << endl;
    }
};

// Utility functions
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
        if (line.empty())
            continue;

        // Skip comment lines or lines that don't start with a digit
        if (line[0] == '#' || !isdigit(line[0]))
            continue;

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
                {
                    e.is_removal = true;
                }
                else
                {
                    if (iss >> weight_str)
                    {
                        // Try to parse as ll
                        e.weight = stoll(weight_str);
                    }
                    
                }

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
    {
        file << i << " " << fixed << setprecision(2);

        if (dist[i] == LLONG_MAX)
            file << "inf\n";
        else
            file << dist[i] << "\n";
    }
}

void printStats(const vector<ll> &dist)
{
    int reachable = 0;
    ll max_dist = 0;
    ll sum_dist = 0;

    for (ll d : dist)
    {
        if (d < LLONG_MAX)
        {
            reachable++;
            if (d > max_dist)
                max_dist = d;
            sum_dist += d;
        }
    }

    cout << "SSSP Statistics:\n";
    cout << "  Reachable vertices: " << reachable << "/" << dist.size() << "\n";
    cout << "  Maximum distance: " << max_dist << "\n";
    cout << "  Average distance: " << (reachable > 0 ? sum_dist / reachable : 0) << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0]
                  << " <graph_file> <updates_file> <source_vertex> [output_file]" << endl;
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
        cerr << "Error: Source vertex must be a valid integer, got '" << argv[3] << "'" << endl;
        return 1;
    }

    // Default values
    string output_file = "";

    // Process optional output file
    if (argc > 4)
    {
        output_file = argv[4];
    }

    cout << "Configuration:" << endl;
    cout << "  Graph file: " << graph_file << endl;
    cout << "  Updates file: " << updates_file << endl;
    cout << "  Source vertex: " << source << endl;
    cout << "  Output file: " << (output_file.empty() ? "none" : output_file) << endl;

    // Load graph
    Graph graph;
    cout << "Loading graph from " << graph_file << endl;
    graph.loadFromFile(graph_file);

    // Validate source vertex
    if (source < 0 || source >= graph.V)
    {
        cerr << "Error: Source vertex " << source << " is out of range (0 to " << (graph.V - 1) << ")" << endl;
        return 1;
    }

    cout << "Graph loaded: " << graph.V << " vertices, " << graph.E << " edges" << endl;

    // Initialize SSSP
    SSSP sssp(graph.V);

    cout << "Running initial SSSP calculation from source " << source << endl;
    sssp.dijkstra(graph, source);

    vector<ll> initial_dist = sssp.dist;
    cout << "Initial SSSP completed. Statistics:" << endl;
    printStats(initial_dist);

    // Load updates
    cout << "Loading updates from " << updates_file << endl;
    vector<UpdateData> all_updates = loadUpdates(updates_file);
    cout << "Loaded " << all_updates.size() << " updates" << endl;

    cout << "Processing " << all_updates.size() << " updates" << endl;

    auto start_time = chrono::high_resolution_clock::now();

    // Apply all updates to the graph structure
    graph.applyUpdates(all_updates);

    // Recompute SSSP from scratch
    sssp.dijkstra(graph, source);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    cout << "SSSP update completed in " << duration << " ms\n";
    printStats(sssp.dist);

    if (!output_file.empty())
    {
        saveResults(output_file, sssp.dist);
        cout << "Results saved to " << output_file << "\n";
    }

    return 0;
}