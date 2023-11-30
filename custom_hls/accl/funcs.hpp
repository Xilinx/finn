#pragma once

#ifdef CPPSIM
#include <iostream>
#endif

const size_t accl_width = 512;

template<unsigned int stream_width, unsigned int num_bits, unsigned int step>
void accl_out(
    unsigned int destination,
    ap_uint<32> comm_adr,
    ap_uint<32> dpcfg_adr,
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    hls::stream<ap_uint<stream_width>> &in,
    bool wait_for_ack
) {
    STREAM<stream_word> data_from_cclo;

    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    ap_uint<accl_width> accl_word;
    ap_uint<stream_width> stream_word;

#ifdef CPPSIM
    std::cerr << "accl_out starting to output data to rank " << destination << " (" << num_bits << " bits)" << std::endl;
#endif

    int num_transfer_bits = ((num_bits + accl_width - 1) / accl_width) * accl_width;

    send: for (int i = 0; i < num_bits - step + 1; i += step) {
        if (i % stream_width == 0) {
            stream_word = in.read();
        }

        int ni = i + step - 1;

        accl_word(ni % accl_width, i % accl_width) =
            stream_word(ni % stream_width, i % stream_width);

        if ((ni + 1) % accl_width == 0) {
            data.push(accl_word, 0);
        }
    }

    if (num_bits < num_transfer_bits) {
        data.push(accl_word, 0);
    }

    accl.stream_put(num_transfer_bits / 32, 9, destination, 0, false);

#ifdef CPPSIM
    std::cerr << "accl_out waiting on ack" << std::endl;
#endif

    if (wait_for_ack) accl.finalize_call();

#ifdef CPPSIM
    std::cerr << "accl_out finished" << std::endl;
#endif
}

template<unsigned int stream_width, unsigned int num_bits, unsigned int step>
void accl_in(
    unsigned int source,
    STREAM<stream_word> &data_from_cclo,
    hls::stream<ap_uint<stream_width>> &out
) {
    STREAM<stream_word> data_to_cclo;
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);

    ap_uint<accl_width> accl_word;
    ap_uint<stream_width> stream_word;

#ifdef CPPSIM
    std::cerr << "accl_in starting to receive data from rank " << source << " (" << num_bits << " bits)" << std::endl;
#endif

    recv: for (int i = 0; i < num_bits - step + 1; i += step) {
        if (i % accl_width == 0) {
            accl_word = data.pull().data;
        }

        int ni = i + step - 1;

        stream_word(ni % stream_width, i % stream_width) =
            accl_word(ni % accl_width, i % accl_width);

        if ((ni + 1) % stream_width == 0) {
            out.write(stream_word);
        }
    }

#ifdef CPPSIM
    std::cerr << "accl_in finished" << std::endl;
#endif
}
