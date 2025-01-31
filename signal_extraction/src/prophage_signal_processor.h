// prophage_signal_processor.h

#ifndef PROPHAGE_SIGNAL_PROCESSOR_H
#define PROPHAGE_SIGNAL_PROCESSOR_H

#include <vector>
#include <string>
#include <utility>

// Add the struct definition here, before the function declarations
struct CSVData {
    std::vector<std::vector<std::string>> rows;  // All data rows
    std::vector<std::string> headers;            // Column headers
    std::vector<int> labels;                     // Predicted labels
    std::vector<int> trueLabels;                // Reference labels
};

// Function declarations
CSVData readCSV(const std::string& filename);
void writeCSV(const std::string& filename, const std::vector<int>& data);
void writeCombinedCSV(const std::string& filename, 
                      const CSVData& originalData,
                      const std::vector<std::pair<std::string, std::vector<int>>>& algorithmResults);

std::vector<int> movingWindowAverage(const std::vector<int>& input, int windowSize, double threshold, int numThreads);
std::vector<int> runLengthEncoding(const std::vector<int>& input, int minLength, int numThreads);
std::vector<int> dbscan(const std::vector<int>& input, int eps, int minPts, int numThreads);
std::vector<int> medianFilter(const std::vector<int>& input, int windowSize, int numThreads);
std::vector<int> connectedComponentLabeling(const std::vector<int>& input, int minSize, int gapTolerance, int numThreads);
std::vector<int> windowSum(const std::vector<int>& input, int windowSize, int threshold, int numThreads);

void calculateMetrics(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, const std::vector<int>& algorithmOutput, const std::string& algorithmName);
void printHelp();

#endif // PROPHAGE_SIGNAL_PROCESSOR_H
