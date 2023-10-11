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
    const std::vector<unsigned int> &dest,
    hlslib::Stream<command_word> &cmd_to_cclo,
    hlslib::Stream<command_word> &sts_from_cclo,
    hlslib::Stream<stream_word> &data_from_cclo,
    hlslib::Stream<stream_word> &data_to_cclo
) {
    auto cclo = std::make_unique<CCLO_BFM>(zmqport, rank, world_size, dest,
                    cmd_to_cclo, sts_from_cclo, data_from_cclo, data_to_cclo);
    cclo->run();

    // Makeshift barrier
    std::cout << "CCLO BFM started" << std::endl;
    std::string inp;
    std::cin >> inp;

    return cclo;
}

template<unsigned int accl_width, unsigned int stream_width, unsigned int count>
void accl_out(
    unsigned int destination,
    ap_uint<32> comm_adr,
    ap_uint<32> dpcfg_adr,
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo,
    hls::stream<ap_uint<stream_width>> &in
) {
    #pragma HLS INTERFACE axis port=cmd_to_cclo
    #pragma HLS INTERFACE axis port=sts_from_cclo
    #pragma HLS INTERFACE axis port=data_to_cclo
    #pragma HLS INTERFACE axis port=data_from_cclo
    #pragma HLS INTERFACE axis port=in

    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    ap_uint<512> accl_word;
    ap_uint<stream_width> stream_word;

    int num_bits = count * accl_width;
    int step = std::gcd(accl_width, stream_width);

    std::cerr << "accl_out starting to output data to rank " << destination << " (" << num_bits << " bits)" << std::endl;

    for (int i = 0; i < num_bits - step + 1; i += step) {
        if (i % stream_width == 0) {
            stream_word = in.read();
        }

        int ni = i + step;

        accl_word(i % accl_width, ni % accl_width) =
            stream_word(i % stream_width, ni % stream_width);

        if (ni % accl_width == 0) {
            data.push(accl_word, 0);
        }
    }

    std::cerr << "accl_out calling accl" << std::endl;

    accl.stream_put(num_bits / 32, 9, destination, (ap_uint<64>)&accl_word);

    std::cerr << "accl_out finished" << std::endl;
}

template<unsigned int accl_width, unsigned int stream_width, unsigned int count>
void accl_in(
    unsigned int source,
    ap_uint<32> comm_adr,
    ap_uint<32> dpcfg_adr,
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo,
    hls::stream<ap_uint<stream_width>> &out
) {
    #pragma HLS INTERFACE axis port=cmd_to_cclo
    #pragma HLS INTERFACE axis port=sts_from_cclo
    #pragma HLS INTERFACE axis port=data_to_cclo
    #pragma HLS INTERFACE axis port=data_from_cclo
    #pragma HLS INTERFACE axis port=out

    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    ap_uint<512> accl_word;
    ap_uint<stream_width> stream_word;

    int num_bits = count * stream_width;
    int step = std::gcd(accl_width, stream_width);

    std::cerr << "accl_in start receiving data (" << num_bits << " bits)" << std::endl;

    for (int i = 0; i < num_bits - step + 1; i += step) {
        if (i % accl_width == 0) {
            accl_word = data.pull().data;
        }

        int ni = i + step;

        stream_word(i % stream_width, ni % stream_width) =
            accl_word(i % accl_width, ni % accl_width);

        if (ni % accl_width == 0) {
            out.write(stream_word);
        }
    }

    std::cerr << "accl_in finished" << std::endl;
}
