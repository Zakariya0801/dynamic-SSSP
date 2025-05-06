#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>

int main() {
    std::string filename = "colisten-spotify.txt"; // Replace with your actual file name
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    int maxNode = std::numeric_limits<int>::min();
    int edgeCount = 0;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int u, v, weight;

        if (!(iss >> u >> v >> weight)) {
            std::cerr << "Error: Invalid line format: " << line << std::endl;
            continue;
        }

        maxNode = std::max({maxNode, u, v});
        edgeCount++;
    }

    file.close();

    std::cout << "Maximum Node Value: " << maxNode << std::endl;
    std::cout << "Number of Edges: " << edgeCount << std::endl;

    return 0;
}