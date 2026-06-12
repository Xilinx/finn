#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <sstream>
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <cctype>

extern "C" {
    #include "dmabwave.h"
}

#include "extract_sys.hpp"

/**
 * Class to manage FPGA resources
 */

class FPGAOp {
    // Init
    bool is_initialized;

    // Sizes
    uint32_t in_size;
    uint32_t out_size;

    // FPGA handle
    volatile uint64_t* csr;

    // Buffers
    std::vector<void*> weight_bufs;

    // Chrono
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

private: 

    // Util
    static std::string upper(std::string s) {
        for (auto& c : s) c = (char)std::toupper((unsigned char)c);
        return s;
    }

    static const char* torch_dtype_label(c10::ScalarType st) {
        switch (st) {
            case torch::kInt8:    return "INT8";
            case torch::kUInt8:   return "UINT8";
            case torch::kInt16:   return "INT16";
            case torch::kFloat32: return "FP32";
            default:              return "OTHER";
        }
    }

    static c10::ScalarType to_torch_dtype(const char* tag) {
        if (!tag) return torch::kUInt8;
        if (std::strcmp(tag,"INT8")==0)    return torch::kInt8;
        if (std::strcmp(tag,"UINT8")==0)   return torch::kUInt8;
        if (std::strcmp(tag,"INT16")==0)   return torch::kInt16;
        if (std::strcmp(tag,"FP32")==0 ||
            std::strcmp(tag,"FLOAT32")==0) return torch::kFloat32;
        return torch::kUInt8; // safe default
    }

    static size_t dtype_bytes(c10::ScalarType dt){
        switch(dt){
            case torch::kInt8:
            case torch::kUInt8:   return 1;
            case torch::kInt16:   return 2;
            case torch::kFloat32: return 4;
            default:              return 1;
        }
    }

public:

    //
    // Ctor
    FPGAOp(std::optional<std::vector<torch::Tensor>> weight_tensors) : 
        is_initialized(false)
    {
        // Dtypes
        constexpr std::size_t EL_BYTES_IN = 
            (std::string_view(G_IN_DTYPE) == "FLOAT32") ? 4 :
            (std::string_view(G_IN_DTYPE) == "INT32")   ? 4 :
            (std::string_view(G_IN_DTYPE) == "INT16")   ? 2 :
            (std::string_view(G_IN_DTYPE) == "UINT16")  ? 2 :
            (std::string_view(G_IN_DTYPE) == "INT8")    ? 1 :
            (std::string_view(G_IN_DTYPE) == "UINT8")   ? 1 :
            throw std::runtime_error("Unsupported G_IN_DTYPE");

        constexpr std::size_t EL_BYTES_OUT = 
            (std::string_view(G_OUT_DTYPE) == "FLOAT32") ? 4 :
            (std::string_view(G_OUT_DTYPE) == "INT32")   ? 4 :
            (std::string_view(G_OUT_DTYPE) == "INT16")   ? 2 :
            (std::string_view(G_OUT_DTYPE) == "UINT16")  ? 2 :
            (std::string_view(G_OUT_DTYPE) == "INT8")    ? 1 :
            (std::string_view(G_OUT_DTYPE) == "UINT8")   ? 1 :
            throw std::runtime_error("Unsupported G_OUT_DTYPE");

        // Setup the FPGA
        std::string f_path = BW_CONFIG_PATH;
        if(setup_qdma(f_path.c_str()) < 0)
            throw std::runtime_error("Platform setup failed");

        // FPGA handle
        csr = map_csr();
        if(!csr)
            throw std::runtime_error("CSR mapping failed");

        std::cout << "Platform setup completed" << std::endl << std::endl;

        // Weight setup
        const auto& weights = *weight_tensors;

        if(CORE_ID_COUNT > 0) {
            if ((int)weights.size() != CORE_ID_COUNT) {
                throw std::runtime_error("weights.size() must equal CORE_ID_COUNT");
            }

            weight_bufs.reserve(CORE_ID_COUNT);
            
            // Load bytes from tensors
            for (int i = 0; i < CORE_ID_COUNT; ++i) {
                const auto& w = weights[i];
                if (!w.is_contiguous())
                    throw std::runtime_error("weight tensor must be contiguous (core idx " + std::to_string(i) + ")");

                // dtype check
                const std::string want = upper(std::string(CORE_DTYPES[i]));
                const char*       got  = torch_dtype_label(w.scalar_type());
                if (want != got) {
                    throw std::runtime_error("dtype mismatch for core idx " + std::to_string(i) +
                                            " (file " + std::string(got) + ", expected " + want + ")");
                }

                const size_t nbytes = (size_t)w.numel() * (size_t)w.element_size();
                void* buf = std::aligned_alloc(4096, nbytes);
                if (!buf) throw std::runtime_error("aligned_alloc failed");
                std::memset(buf, 0, nbytes);
                weight_bufs.push_back(buf);

                // Byte view and copy
                auto bytes = torch::from_blob(w.data_ptr(), { (long)nbytes }, torch::kUInt8);
                std::memcpy(weight_bufs[i], bytes.data_ptr<uint8_t>(), nbytes);

                // Offload
                int curr_bank = N_HOST_BANKS + N_INTF_BANKS + i;
                start_time = std::chrono::high_resolution_clock::now();
                dma_xfer(q_info->q_name, reinterpret_cast<char*>(weight_bufs[i]), nbytes, BANK[curr_bank], H2C_TRANSFER);
                end_time = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                std::cout << "[OFFLOAD] weights [bank " << i << "] offloaded in: " << duration << " us" << std::endl;
            }
        }

        // Setup CSR
        csr[(int)RegMap::CH_CNFG_RD_ADDR_OFFS] = 0;
        csr[(int)RegMap::CH_CNFG_WR_ADDR_OFFS] = 0;

        in_size = G_IN_LEN * EL_BYTES_IN;
        out_size = G_OUT_LEN * EL_BYTES_OUT;
        csr[(int)RegMap::CH_CNFG_RD_LEN_OFFS] = in_size;
        csr[(int)RegMap::CH_CNFG_WR_LEN_OFFS] = out_size;

        is_initialized = true;
    }

    //
    // Dtor
    ~FPGAOp()
    {
        if (is_initialized)
        {
            std::cout << "Cleaning up resources" << std::endl;

            // Clean the platform
            clean_qdma();
        }
    }

    // Process the input tensor
    torch::Tensor forward(const int batch_size, torch::Tensor user)
    {
        // Init
        if (!is_initialized)
            throw std::runtime_error("FPGA is not initialized");

        // Check and offload user embeddings
        auto chost = user.contiguous();
        const std::size_t nbytes_in = static_cast<std::size_t>(chost.numel()) * chost.element_size();
        if(nbytes_in != batch_size * in_size) {
            std::cout << "nbytes_in: " << nbytes_in << ", int: " << (static_cast<std::size_t>(batch_size) * in_size) << std::endl;
            throw std::runtime_error("Wrong input tensor provided");
        }

        auto chost_view = torch::from_blob(chost.data_ptr(), { (long)nbytes_in }, torch::kUInt8);
        uint8_t* chost_ptr = chost_view.data_ptr<uint8_t>();

        start_time = std::chrono::high_resolution_clock::now();
        dma_xfer(q_info->q_name, reinterpret_cast<char*>(chost_ptr), nbytes_in, BANK[0], H2C_TRANSFER);
        end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "[OFFLOAD] input offloaded in: " << duration << " us" << std::endl << std::endl;

        // Run
        std::cout << "[RUN] starting ... (batch size " << batch_size << ")" << std::endl;
        csr[(int)RegMap::N_RUNS_REG] = batch_size;
        csr[(int)RegMap::CTRL_REG] = 0x1;

        while((csr[(int)RegMap::CTRL_REG] & 0x1) != 0x1) { }
        std::cout << "[RUN] completed!" << std::endl;

        
        // Sync results
        const auto out_dt   = to_torch_dtype(G_OUT_DTYPE);
        const int64_t elems_per_sample = G_OUT_LEN;

        auto out = torch::empty({static_cast<int64_t>(batch_size), elems_per_sample}, 
            torch::TensorOptions().dtype(out_dt).device(torch::kCPU));
        auto* out_ptr = reinterpret_cast<char*>(out.data_ptr());

        start_time = std::chrono::high_resolution_clock::now();
        dma_xfer(q_info->q_name, out_ptr, static_cast<size_t>(batch_size) * out_size, BANK[1], C2H_TRANSFER);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "[SYNC] results synced in: " << duration << " us" << std::endl << std::endl;

        // Performance readout
        uint64_t lint;
	    double qps;
        double sclk = 4.0; // System clock
        std::cout << "[PERF] Total latency: " << csr[(int)RegMap::PERF_LAT_REG] << " cycles (" << ((csr[(int)RegMap::PERF_LAT_REG] * sclk) / 1e6) << " ms)" << std::endl;
        for(int i = 0; i < batch_size; i++) {
            if(i!=batch_size-1) { 
                std::cout << "   ├── Interval (fpga_batch) " << i << ": " << csr[(int)RegMap::PERF_INT_REG] << " cycles (" << ((csr[(int)RegMap::PERF_INT_REG] * sclk) / 1e6) << " ms)" << std::endl;
            } else {
                lint = csr[(int)RegMap::PERF_INT_REG];
                qps = (batch_size * 1e9) / (lint * sclk);
                std::cout << "   └── Interval (fpga_batch) " << i << ": " << lint << " cycles (" << ((lint * sclk) / 1e6) << " ms)" << std::endl;
            }		    
            csr[(int)RegMap::PERF_INT_REG] = 0x1; // pop
        }

        std::cout << "[QPS] last iteration: " << (double) (qps * (1024.0/1000)) << std::endl << std::endl;

        return out;
    }

    // Getter for initialization status
    bool is_initialized_status() const
    {
        return is_initialized;
    }

    // Expose constants
    static std::vector<int>      py_core_ids()   { return {CORE_IDS.begin(), CORE_IDS.end()}; }
    static std::vector<uint32_t> py_core_mhmw()  { return {CORE_MHMW.begin(), CORE_MHMW.end()}; }
    static std::vector<std::string> py_core_dtypes(){
        std::vector<std::string> out; out.reserve(CORE_ID_COUNT);
        for (int i = 0; i < CORE_ID_COUNT; ++i) out.emplace_back(CORE_DTYPES[i]);
            return out;
    }
    static int py_core_id_count() { return CORE_ID_COUNT; }

};

//
// Module definitions
//

PYBIND11_MODULE(brainwave, m) {
    auto cls = py::class_<FPGAOp>(m, "FPGAOp")
        .def(py::init<std::optional<std::vector<torch::Tensor>>>(),
             py::arg("weight_tensors") = py::none())
        .def("forward", &FPGAOp::forward)
        .def("is_initialized", &FPGAOp::is_initialized_status)
        .def_static("in_dtype",  []{ return std::string(G_IN_DTYPE); })
        .def_static("out_dtype", []{ return std::string(G_OUT_DTYPE); })
        .def_static("core_id_count", &FPGAOp::py_core_id_count)
        .def_static("core_ids",      &FPGAOp::py_core_ids)
        .def_static("core_mhmw",     &FPGAOp::py_core_mhmw)
        .def_static("core_dtypes",   &FPGAOp::py_core_dtypes);
}