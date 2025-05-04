#ifndef UTILS_H
#define UTILS_H
#include "graph.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

using namespace std;

vector<UpdateData> loadUpdates(const string &filename);
void saveResults(const string &filename, const vector<ll> &dist);
void printStats(const vector<ll> &dist);

#endif // UTILS_H