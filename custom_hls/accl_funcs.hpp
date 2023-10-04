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

    for (int i = 0; i < num_bits - step + 1; num_bits += step) {
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

    accl.stream_put(num_bits / 32, 9, destination, 0);
}

template<unsigned int accl_width, unsigned int stream_width, unsigned int count>
void accl_in(
    unsigned int destination,
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

    int num_bits = count * accl_width;
    int step = std::gcd(accl_width, stream_width);

    for (int i = 0; i < num_bits - step + 1; num_bits += step) {
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
}
