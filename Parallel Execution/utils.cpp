#include "utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>

using namespace std;
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