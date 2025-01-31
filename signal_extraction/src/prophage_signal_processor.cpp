// prophage_signal_processor.cpp

#include "prophage_signal_processor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    try {
        // Read the input file
        auto csvData = readCSV(inputFile);
        std::vector<std::pair<std::string, std::vector<int>>> allResults;

        // Define algorithm parameters
        struct AlgorithmParams {
            std::string name;
            std::vector<std::string> params;
        };

        std::vector<AlgorithmParams> algorithms = {
            {"mwa", {"70", "0.2"}},
            {"rle", {"8"}},
            {"dbscan", {"50", "20"}},
            {"median", {"50"}},
            {"ccl", {"40", "8"}},
            {"window_sum", {"85", "25"}}
        };

        // Process each algorithm
        for (const auto& algo : algorithms) {
            std::vector<int> result;
            try {
                if (algo.name == "mwa") {
                    result = movingWindowAverage(csvData.labels, std::stoi(algo.params[0]), 
                                              std::stod(algo.params[1]), 4);
                }
                else if (algo.name == "rle") {
                    result = runLengthEncoding(csvData.labels, std::stoi(algo.params[0]), 4);
                }
                else if (algo.name == "dbscan") {
                    result = dbscan(csvData.labels, std::stoi(algo.params[0]), 
                                  std::stoi(algo.params[1]), 4);
                }
                else if (algo.name == "median") {
                    result = medianFilter(csvData.labels, std::stoi(algo.params[0]), 4);
                }
                else if (algo.name == "ccl") {
                    result = connectedComponentLabeling(csvData.labels, std::stoi(algo.params[0]), 
                                                      std::stoi(algo.params[1]), 4);
                }
                else if (algo.name == "window_sum") {
                    result = windowSum(csvData.labels, std::stoi(algo.params[0]), 
                                     std::stoi(algo.params[1]), 4);
                }

                if (!result.empty()) {
                    calculateMetrics(csvData.trueLabels, csvData.labels, result, algo.name);
                    allResults.emplace_back(algo.name, result);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing algorithm " << algo.name << ": " << e.what() << "\n";
            }
        }

        // Write combined results
        writeCombinedCSV(outputFile, csvData, allResults);
        std::cout << "Processing complete. Combined results written to " << outputFile << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

std::vector<int> windowSum(const std::vector<int>& input, int windowSize, int threshold, int numThreads) {
    std::vector<int> output(input.size(), 0);
    std::vector<std::thread> threads;
    
    auto worker = [&](int start, int end) {
        int sum = 0;
        // Initialize sum for first window in this thread's section
        for (int i = std::max(0, start - windowSize + 1); i <= start; ++i) {
            if (i >= 0) sum += input[i];
        }
        
        for (int i = start; i < end; ++i) {
            // Add new value to sum
            if (i < input.size()) {
                sum += input[i];
            }
            
            // Remove value outside window
            if (i >= windowSize) {
                sum -= input[i - windowSize];
            }
            
            // If we have a full window, check against threshold
            if (i >= windowSize - 1) {
                output[i - windowSize + 1] = (sum >= threshold) ? 1 : 0;
            }
        }
    };

    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

std::vector<int> movingWindowAverage(const std::vector<int>& input, int windowSize, double threshold, int numThreads) {
    std::vector<int> output(input.size(), 0);
    std::vector<std::thread> threads;
    
    auto worker = [&](int start, int end) {
        int sum = 0;
        for (int i = start; i < end; ++i) {
            sum += input[i];
            if (i >= windowSize) sum -= input[i - windowSize];
            
            if (i >= windowSize - 1) {
                double avg = static_cast<double>(sum) / windowSize;
                output[i - windowSize / 2] = (avg >= threshold) ? 1 : 0;
            }
        }
    };

    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

std::vector<int> runLengthEncoding(const std::vector<int>& input, int minLength, int numThreads) {
    std::vector<int> output(input.size(), 0);
    std::vector<std::pair<int, int>> runs;
    std::mutex runsMutex;

    auto worker = [&](int start, int end) {
        int count = 0;
        int runStart = -1;
        
        for (int i = start; i < end; ++i) {
            if (input[i] == 1) {
                if (count == 0) runStart = i;
                count++;
            } else {
                if (count >= minLength) {
                    std::lock_guard<std::mutex> lock(runsMutex);
                    runs.emplace_back(runStart, i);
                }
                count = 0;
            }
        }
        
        if (count >= minLength) {
            std::lock_guard<std::mutex> lock(runsMutex);
            runs.emplace_back(runStart, end);
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (const auto& run : runs) {
        std::fill(output.begin() + run.first, output.begin() + run.second, 1);
    }

    return output;
}

std::vector<int> dbscan(const std::vector<int>& input, int eps, int minPts, int numThreads) {
    std::vector<int> output(input.size(), 0);
    std::vector<std::atomic<bool>> visited(input.size());
    std::vector<std::mutex> pointMutexes(input.size());

    auto expandCluster = [&](int point) {
        std::queue<int> seeds;
        seeds.push(point);

        while (!seeds.empty()) {
            int currentPoint = seeds.front();
            seeds.pop();

            if (!visited[currentPoint].load()) {
                visited[currentPoint].store(true);
                if (input[currentPoint] == 1) {
                    std::lock_guard<std::mutex> lock(pointMutexes[currentPoint]);
                    output[currentPoint] = 1;

                    for (int i = std::max(0, currentPoint - eps); i < std::min(static_cast<int>(input.size()), currentPoint + eps + 1); ++i) {
                        if (!visited[i].load()) seeds.push(i);
                    }
                }
            }
        }
    };

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            if (!visited[i].load() && input[i] == 1) {
                int count = 0;
                for (int j = std::max(0, i - eps); j < std::min(static_cast<int>(input.size()), i + eps + 1); ++j) {
                    if (input[j] == 1) count++;
                }

                if (count >= minPts) {
                    expandCluster(i);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

std::vector<int> medianFilter(const std::vector<int>& input, int windowSize, int numThreads) {
    std::vector<int> output(input.size());

    auto worker = [&](int start, int end) {
        std::vector<int> window(windowSize);
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < windowSize; ++j) {
                int idx = i - windowSize / 2 + j;
                window[j] = (idx >= 0 && idx < input.size()) ? input[idx] : 0;
            }

            std::sort(window.begin(), window.end());
            output[i] = window[windowSize / 2];
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

std::vector<int> connectedComponentLabeling(const std::vector<int>& input, int minSize, int gapTolerance, int numThreads) {
    std::vector<int> output(input.size(), 0);
    std::vector<int> labels(input.size(), 0);
    std::atomic<int> currentLabel(1);

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            if (input[i] == 1 && labels[i] == 0) {
                int componentStart = i;
                int label = currentLabel.fetch_add(1);
                
                // Forward pass: label and expand
                while (i < end) {
                    labels[i] = label;
                    ++i;
                    // Check for gap
                    int gapSize = 0;
                    while (i < end && input[i] == 0 && gapSize < gapTolerance) {
                        labels[i] = label;
                        ++i;
                        ++gapSize;
                    }
                    if (i < end && input[i] == 1) {
                        continue; // Continue expanding
                    } else {
                        break; // End of component
                    }
                }
                
                // Backward pass: expand in reverse direction
                for (int j = componentStart - 1; j >= start && componentStart - j <= gapTolerance; --j) {
                    if (input[j] == 1) {
                        componentStart = j;
                        break;
                    }
                    labels[j] = label;
                }
                
                // Check component size and set output
                int componentSize = i - componentStart;
                if (componentSize >= minSize) {
                    std::fill(output.begin() + componentStart, output.begin() + i, 1);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = input.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? input.size() : (i + 1) * chunkSize;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

CSVData readCSV(const std::string& filename) {
    CSVData data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Read header
    if (std::getline(file, line)) {
        std::istringstream headerStream(line);
        std::string column;
        while (std::getline(headerStream, column, ',')) {
            data.headers.push_back(column);
        }
        
        // Verify we have enough columns
        if (data.headers.size() < 7) {
            throw std::runtime_error("CSV file does not have enough columns");
        }
    }

    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> row;
        
        while (std::getline(iss, token, ',')) {
            row.push_back(token);
        }
        
        if (row.size() >= 7) {
            data.rows.push_back(row);
            try {
                data.labels.push_back(std::stoi(row[5]));
                data.trueLabels.push_back(std::stoi(row[6]));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Error parsing values: " << e.what() << "\n";
                continue;
            }
        }
    }

    if (data.rows.empty()) {
        throw std::runtime_error("No valid data was read from the file");
    }

    return data;
}

void writeCSV(const std::string& filename, const std::vector<int>& data) {
    std::ofstream file(filename);
    file << "label\n";
    for (int value : data) {
        file << value << "\n";
    }
}

void writeCombinedCSV(const std::string& filename, 
                      const CSVData& originalData,
                      const std::vector<std::pair<std::string, std::vector<int>>>& algorithmResults) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    // Write headers
    for (size_t i = 0; i < originalData.headers.size(); ++i) {
        file << originalData.headers[i];
        if (i < originalData.headers.size() - 1 || !algorithmResults.empty()) {
            file << ",";
        }
    }
    // Add algorithm result headers
    for (size_t i = 0; i < algorithmResults.size(); ++i) {
        file << algorithmResults[i].first;
        if (i < algorithmResults.size() - 1) {
            file << ",";
        }
    }
    file << "\n";

    // Write data rows
    for (size_t i = 0; i < originalData.rows.size(); ++i) {
        // Write original data
        for (size_t j = 0; j < originalData.rows[i].size(); ++j) {
            file << originalData.rows[i][j];
            if (j < originalData.rows[i].size() - 1 || !algorithmResults.empty()) {
                file << ",";
            }
        }
        // Write algorithm results
        for (size_t j = 0; j < algorithmResults.size(); ++j) {
            file << algorithmResults[j].second[i];
            if (j < algorithmResults.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
}

void calculateMetrics(const std::vector<int>& trueLabels, 
                     const std::vector<int>& predictedLabels, 
                     const std::vector<int>& algorithmOutput,
                     const std::string& algorithmName) {
    auto calculateMetricsForLabels = [](const std::vector<int>& true_labels, 
                                      const std::vector<int>& pred_labels) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (true_labels[i] == 1 && pred_labels[i] == 1) tp++;
            else if (true_labels[i] == 0 && pred_labels[i] == 1) fp++;
            else if (true_labels[i] == 0 && pred_labels[i] == 0) tn++;
            else if (true_labels[i] == 1 && pred_labels[i] == 0) fn++;
        }

        double accuracy = static_cast<double>(tp + tn) / (tp + tn + fp + fn);
        double precision = tp > 0 ? static_cast<double>(tp) / (tp + fp) : 0;
        double recall = tp > 0 ? static_cast<double>(tp) / (tp + fn) : 0;
        double f1 = (precision + recall) > 0 ? 
                   2 * (precision * recall) / (precision + recall) : 0;
        
        double mcc_numerator = static_cast<double>(tp * tn - fp * fn);
        double mcc_denominator = std::sqrt(static_cast<double>(tp + fp) * 
                                         (tp + fn) * (tn + fp) * (tn + fn));
        double mcc = mcc_denominator > 0 ? mcc_numerator / mcc_denominator : 0;

        return std::make_tuple(accuracy, precision, recall, f1, mcc);
    };

    std::cout << "Metrics for " << algorithmName << ":\n";
    auto [accuracy, precision, recall, f1, mcc] = 
        calculateMetricsForLabels(trueLabels, algorithmOutput);
    
    std::cout << "Accuracy: " << accuracy << "\n"
              << "Precision: " << precision << "\n"
              << "Recall: " << recall << "\n"
              << "F1 Score: " << f1 << "\n"
              << "MCC: " << mcc << "\n\n";
}

void printHelp() {
     std::cout << "Usage: prophage_signal_processor <input_file> <output_file> <algorithm> [parameters] [num_threads]\n\n"
              << "Algorithms:\n"
              << "  mwa <window_size> <threshold>   : Moving Window Average\n"
              << "  rle <min_length>                : Run Length Encoding\n"
              << "  dbscan <eps> <min_pts>          : DBSCAN\n"
              << "  median <window_size>            : Median Filter\n"
              << "  ccl <min_size>                  : Connected Component Labeling\n"
              << "  window_sum <window_size> <threshold> : Window Sum\n\n"
              << "Example:\n"
              << "  prophage_signal_processor input.csv output.csv mwa 5 0.6 4\n";
}
