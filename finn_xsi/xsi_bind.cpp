/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Python binding for FINN XSI++.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

#include <pybind11/pybind11.h>
#include "xsi_finn.hpp"
#include <mutex>
#include <map>


namespace  py = pybind11;
using namespace xsi;

namespace {
	std::mutex                                              use_mutex;
	std::map<Design const*, std::shared_ptr<Kernel> const>  use_map;
	struct DesignDeleter : public std::default_delete<Design> {
		void operator()(Design *d) const {
			std::default_delete<Design>::operator()(d);
			std::lock_guard<std::mutex>  lock(use_mutex);
			use_map.erase(use_map.find(d));
		}
	};
}

PYBIND11_MODULE(xsi, m) {

	py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel")
		.def(py::init<std::string const&>())
		.def("hex_in_lower", &Kernel::hex_in_lower)
		.def("hex_in_upper", &Kernel::hex_in_upper);

	py::class_<Design, std::unique_ptr<Design, DesignDeleter>>(m, "Design")
		.def(py::init([](
			std::shared_ptr<Kernel> const &kernel,
			std::string const &design_lib,
			char const *const  log_file,
			char const *const  wdb_file
		) {
			std::unique_ptr<Design, DesignDeleter>  d { new Design(*kernel, design_lib, log_file, wdb_file) };
			std::lock_guard<std::mutex>  lock(use_mutex);
			use_map.emplace(d.get(), kernel);
			return  d;
		}))
		.def("trace_all", &Design::trace_all)
		.def("run",       &Design::run)
		.def("restart",   &Design::restart)
		.def("get_status",     &Design::get_status)
		.def("get_error_info", &Design::get_error_info)
		.def("num_ports",      &Design::num_ports)
		.def("getPort",        static_cast<Port* (Design::*)(std::string const&)>(&Design::getPort))
		.def("ports", [](Design &d) {
			auto const  e = d.ports();
			return  py::make_iterator(e.begin(), e.end());
		});

	py::class_<Port, std::unique_ptr<Port, py::nodelete>>(m, "Port")
		.def("name",        &Port::name)
		.def("dir",         &Port::dir)
		.def("width",       &Port::width)
		.def("isInput",     &Port::isInput)
		.def("isOutput",    &Port::isOutput)
		.def("isInout",     &Port::isInout)
		.def("read",        &Port::read)
		.def("write_back",  &Port::write_back)
		.def("hasUnknown",  &Port::hasUnknown)
		.def("isZero",      &Port::isZero)
		.def("as_bool",     &Port::as_bool)
		.def("as_unsigned", &Port::as_unsigned)
		.def("as_binstr",   &Port::as_binstr)
		.def("as_hexstr",   &Port::as_hexstr)
		.def("clear",       &Port::clear)
		.def("set",         &Port::set)
		.def("set_binstr",  &Port::set_binstr)
		.def("set_hexstr",  &Port::set_hexstr);
}
