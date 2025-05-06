#include <iostream>
#include <fstream>
#include <string>

using namespace std;
int countLinesInFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return -1;
    }
    ofstream outfile("colisten-spotify.txt");
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return -1;
    }
    
    unsigned long long lineCount = 0;
    std::string line;
    unsigned long long maxLines = 10000000;
    while (std::getline(file, line)) {
        if(maxLines == 0)break;
        maxLines--;
        ++lineCount;
        outfile << line << '\n';
    }
    outfile.close();
    file.close();
    return lineCount;
}
// /media/zakariya/My\ Files/FAST/Semester6/PDC/Project/Code
int main() {
    std::string filename = "/media/zakariya/My Files/FAST/Semester6/PDC/Project/Code/colisten-Spotify.txt";

    int lineCount = countLinesInFile(filename);
    if (lineCount != -1) {
        std::cout << "The file " << filename << " has " << lineCount << " lines." << std::endl;
    }

    return 0;
}