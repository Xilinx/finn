// PLACEHOLDER FOR TESTING
#include <string>
#include <array>

std::string PLATFORM = "alveo";
std::string TRANSFER_MODE = "memory_buffered";

std::array<std::string, 3> IDMA_NAMES = {"a", "b", "c"};
std::array<int, 3> ISHAPE_NORMAL = {1, 2, 3};
std::array<int, 3> ISHAPE_FOLDED = {1, 2, 3};
std::array<int, 3> ISHAPE_PACKED = {1, 2, 3};

std::array<std::string, 3> ODMA_NAMES = {"a", "b", "c"};
std::array<int, 3> OSHAPE_NORMAL = {1, 2, 3};
std::array<int, 3> OSHAPE_FOLDED = {1, 2, 3};
std::array<int, 3> OSHAPE_PACKED = {1, 2, 3};