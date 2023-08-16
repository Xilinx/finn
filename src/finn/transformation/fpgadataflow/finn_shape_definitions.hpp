// PLACEHOLDER FOR TESTING
#include <string>
#include <vector>

std::string PLATFORM = "alveo";
std::string TRANSFER_MODE = "memory_buffered";

std::vector<int> INPUT_BYTEWIDTH = {1, 1, 2};
std::vector<int> OUTPUT_BYTEWIDTH = {1, 1, 2};

std::vector<std::string> IDMA_NAMES = {"a", "b", "c"};
std::vector<std::vector<int>> ISHAPE_NORMAL = {{1,2,3}};
std::vector<std::vector<int>> ISHAPE_FOLDED = {{1,2,3}};
std::vector<std::vector<int>> ISHAPE_PACKED = {{1,2,3}};

std::vector<std::string> ODMA_NAMES = {"a", "b", "c"};
std::vector<std::vector<int>> OSHAPE_NORMAL = {{1,2,3}};
std::vector<std::vector<int>> OSHAPE_FOLDED = {{1,2,3}};
std::vector<std::vector<int>> OSHAPE_PACKED = {{1,2,3}};