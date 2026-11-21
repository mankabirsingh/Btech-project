#include <iostream>
#include <vector>
#include <cstring>
#include <memory>

class DataProcessor {
public:
    // Safe modern C++ method
    std::vector<int> square_elements(const std::vector<int>& input) {
        std::vector<int> result;
        result.reserve(input.size());
        for (int v : input) {
            result.push_back(v * v);
        }
        return result;
    }

    // Vulnerable legacy parsing method with heap buffer overflow
    void parse_legacy_packet(const char* packet_data, size_t length) {
        char* heap_buffer = (char*)malloc(128);
        if (!heap_buffer) return;
        // Insecure memory copy without boundary verification (CWE-122)
        memcpy(heap_buffer, packet_data, length);
        std::cout << "Packet processed." << std::endl;
        free(heap_buffer);
    }
};

int main() {
    DataProcessor proc;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    auto squared = proc.square_elements(numbers);
    return 0;
}
