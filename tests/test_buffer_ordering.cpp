#include "../include/processor.h"
#include "../include/processortool.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <set>
#include <map>
#include "test_utils.h"

const ope::Backend TEST_BACKEND = ope::Backend::CUDA;
const int NUM_FRAMES = 20000;
const int CONTEXT_SIZE = 3;  // Show this many IDs before/after each mismatch

// Tool that tracks both input and output buffer ordering
class BufferOrderingTool : public ope::ProcessorTool {
public:
    std::mutex inputMutex;
    std::mutex outputMutex;
    std::vector<uint64_t> inputIds;   // IDs in submission order (from input callback)
    std::vector<uint64_t> outputIds;  // IDs in receive order (from output callback)
    std::atomic<int> inputCount{0};
    std::atomic<int> outputCount{0};

protected:
    void configureCallbacks() override {
        if (!processor) return;

        // Input callback - records submission order
        rawCallbackId = processor->addInputCallback(
            [this](const ope::IOBuffer& buf) {
                std::lock_guard<std::mutex> lock(inputMutex);
                inputIds.push_back(buf.getBufferId());
                inputCount++;
            }
        );

        // Output callback - records receive order
        processedCallbackId = processor->addOutputCallback(
            [this](const ope::IOBuffer& buf) {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputIds.push_back(buf.getBufferId());
                outputCount++;
            }
        );
    }
};

void printContext(const std::vector<uint64_t>& ids, size_t pos, const std::string& label) {
    size_t start = (pos >= CONTEXT_SIZE) ? pos - CONTEXT_SIZE : 0;
    size_t end = std::min(pos + CONTEXT_SIZE + 1, ids.size());

    std::cout << "    " << label << "[" << start << ".." << end-1 << "]: ";
    for (size_t i = start; i < end; i++) {
        if (i == pos) std::cout << "[";
        std::cout << ids[i];
        if (i == pos) std::cout << "]";
        if (i < end - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

void testBufferOrdering() {
    std::cout << "Testing buffer ordering with " << NUM_FRAMES << " frames..." << std::endl;

    ope::Processor processor(TEST_BACKEND);
    processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
    processor.initialize();

    BufferOrderingTool tool;
    tool.attachToProcessor(&processor);

    // Submit all frames
    for (int i = 0; i < NUM_FRAMES; i++) {
        auto& inputBuffer = processor.getNextAvailableInputBuffer();
        //sleep to simulate acquisition delay
		//std::this_thread::sleep_for(std::chrono::nanoseconds(350000));
        processor.process(inputBuffer);
    }

    // Wait for all callbacks
    while (tool.outputCount < NUM_FRAMES) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Input callbacks received: " << tool.inputIds.size() << std::endl;
    std::cout << "  Output callbacks received: " << tool.outputIds.size() << std::endl;

    // Check for duplicates in output
    std::map<uint64_t, int> outputIdCounts;
    for (uint64_t id : tool.outputIds) {
        outputIdCounts[id]++;
    }

    int duplicateCount = 0;
    for (const auto& pair : outputIdCounts) {
        if (pair.second > 1) {
            duplicateCount++;
            if (duplicateCount <= 5) {
                std::cout << "  [DUPLICATE] Output ID " << pair.first
                          << " received " << pair.second << " times" << std::endl;
            }
        }
    }
    if (duplicateCount > 5) {
        std::cout << "  ... and " << (duplicateCount - 5) << " more duplicates" << std::endl;
    }

    // Check for missing IDs (gaps)
    std::set<uint64_t> expectedIds(tool.inputIds.begin(), tool.inputIds.end());
    std::set<uint64_t> receivedIds(tool.outputIds.begin(), tool.outputIds.end());

    std::vector<uint64_t> missingIds;
    for (uint64_t id : expectedIds) {
        if (receivedIds.find(id) == receivedIds.end()) {
            missingIds.push_back(id);
        }
    }

    if (!missingIds.empty()) {
        std::cout << "  [MISSING] " << missingIds.size() << " IDs not received: ";
        for (size_t i = 0; i < std::min(missingIds.size(), size_t(10)); i++) {
            std::cout << missingIds[i];
            if (i < std::min(missingIds.size(), size_t(10)) - 1) std::cout << ", ";
        }
        if (missingIds.size() > 10) std::cout << "...";
        std::cout << std::endl;
    }

    // Check ordering with context
    std::cout << "\n=== Order Comparison ===" << std::endl;
    int outOfOrderCount = 0;
    int printedMismatches = 0;
    size_t minSize = std::min(tool.inputIds.size(), tool.outputIds.size());

    for (size_t i = 0; i < minSize; i++) {
        if (tool.inputIds[i] != tool.outputIds[i]) {
            outOfOrderCount++;
            if (printedMismatches < 5) {
                std::cout << "  Mismatch at position " << i << ":" << std::endl;
                printContext(tool.inputIds, i, "input ");
                printContext(tool.outputIds, i, "output");
                std::cout << std::endl;
                printedMismatches++;
            }
        }
    }

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "  Total frames: " << NUM_FRAMES << std::endl;
    std::cout << "  Duplicates: " << duplicateCount << std::endl;
    std::cout << "  Missing: " << missingIds.size() << std::endl;
    std::cout << "  Out-of-order positions: " << outOfOrderCount << std::endl;

    if (outOfOrderCount > 0 || duplicateCount > 0 || !missingIds.empty()) {
        std::cout << "  [WARNING] Buffer ordering issues detected!" << std::endl;
    } else {
        std::cout << "  [OK] All frames received in correct order" << std::endl;
    }
}

int main() {
    std::cout << "=== Buffer Ordering Test ===" << std::endl;
    try {
        testBufferOrdering();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
