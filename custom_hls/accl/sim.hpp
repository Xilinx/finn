#include <iostream>
#include <memory>

#include <accl.hpp>
#include <accl_network_utils.hpp>

#include "cclo_bfm.h"

std::unique_ptr<ACCL::ACCL> init_accl(
    unsigned int world_size,
    unsigned int rank,
    unsigned int start_port
) {
    accl_network_utils::acclDesign design = accl_network_utils::acclDesign::AXIS3x;

    std::vector<ACCL::rank_t> ranks;
    // TODO: Get the rxbuf size as a config parameter
    ranks = accl_network_utils::generate_ranks(true, rank, world_size, start_port, 16 * 1024);

    return accl_network_utils::initialize_accl(ranks, rank, true, design);
}

std::unique_ptr<CCLO_BFM> init_cclo_and_wait_for_input(
    unsigned int zmqport,
    unsigned int rank,
    unsigned int world_size,
    hlslib::Stream<command_word> &cmd_to_cclo,
    hlslib::Stream<command_word> &sts_from_cclo,
    hlslib::Stream<stream_word> &data_from_cclo,
    hlslib::Stream<stream_word> &data_to_cclo
) {
    std::vector<unsigned int> dest{9};

    auto cclo = std::make_unique<CCLO_BFM>(zmqport, rank, world_size, dest,
                    cmd_to_cclo, sts_from_cclo, data_from_cclo, data_to_cclo);
    cclo->run();

    // Makeshift barrier
    std::cout << "CCLO BFM started" << std::endl;
    std::string inp;
    std::cin >> inp;

    return cclo;
}


