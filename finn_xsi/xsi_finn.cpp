/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FINN XSI++: C++ XSI Binding used by FINN.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

#include "xsi_finn.hpp"

#include <iostream>
#include <algorithm>


using namespace xsi;

//===========================================================================
// Local Helpers

namespace {
	void* resolve_or_throw(SharedLibrary &lib, char const *const  sym) {
		auto const  res = lib.getsymbol(sym);
		if(!res) {
			throw  std::runtime_error(
				std::string("Failed to resolve ")
				.append(sym).append(" in ").append(lib.path())
			);
		}
		return *res;
	}
	char  XZ10[4] = { '0', '1', 'Z', 'X' };
	char  HEX[16] = {
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'A', 'B', 'C', 'D', 'E', 'F'
	};
}

void Kernel::hex_in_lower() {
	for(unsigned  i =  2; i <  4; i++)  XZ10[i] |=  ' ';
	for(unsigned  i = 10; i < 16; i++)  HEX [i] |=  ' ';
}
void Kernel::hex_in_upper() {
	for(unsigned  i =  2; i <  4; i++)  XZ10[i] &= ~' ';
	for(unsigned  i = 10; i < 16; i++)  HEX [i] &= ~' ';
}

//===========================================================================
// Shared Library Representation

char const SharedLibrary::library_suffix[] =
#if defined(_WIN32)
	".lib";
#else
	".so";
#endif

#if defined(_WIN32)
namespace {
	std::string translate_error_message(DWORD  errid) {
		std::string  msg;
		LPTSTR  bufptr;
		FormatMessage(
			FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr,
			errid,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			&bufptr,
			0, nullptr
		);
		if(bufptr)  msg = reinterpret_cast<char*>(bufptr);
		LocalFree(bufptr);
		return  msg;
	}
}
#endif

SharedLibrary& SharedLibrary::open(std::string const &path) {
	if(_lib)  throw  std::runtime_error("SharedLibrary still open for " + _path);
	_lib  = load(path);
	_path = path;
	return *this;
}

SharedLibrary::handle_type SharedLibrary::load(std::string const &path) {
	if(path.empty())  throw  std::domain_error("Empty library path.");

#if defined(_WIN32)
	SetLastError(0);
#ifdef UNICODE
	// Use LoadLibraryA explicitly on windows if UNICODE is defined
	handle_type const  lib = LoadLibraryA(path.c_str());
#else
	handle_type const  lib = LoadLibrary(path.c_str());
#endif
	if(!lib)  throw  std::runtime_error(translate_error_message(GetLastError()));
#else
	handle_type const  lib = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
	if(!lib)  throw  std::runtime_error(dlerror());
#endif
	return  lib;
}

void SharedLibrary::unload() {
	if(_lib) {
#if defined(_WIN32)
		FreeLibrary(_lib);
#else
		dlclose(_lib);
#endif
	}
}

std::optional<void*> SharedLibrary::getsymbol(char const *const  name) {
	void *sym;
#if defined(_WIN32)
	sym = (void*)GetProcAddress(_lib, name);
	if(!sym)
#else
	dlerror(); // clear error
	sym = dlsym(_lib, name);
	char const *const  err = dlerror();
	if(err)
#endif
		return  std::nullopt;
	return  std::make_optional(sym);
}

//===========================================================================
// xsi::Kernel

char const *const  Kernel::Xsi::FUNC_NAMES[EXTENT] = {
	"xsi_get_value", "xsi_put_value",
	"xsi_get_int_port", "xsi_get_str_port",

	"xsi_get_int", "xsi_get_port_number",

	"xsi_trace_all", "xsi_run", "xsi_restart",
	"xsi_get_status", "xsi_get_error_info",

	"xsi_close"
};

#include <iostream>
inline Kernel::Xsi::Xsi(SharedLibrary &lib) : _hdl(nullptr) {
	// Resolve XSI Functions
	for(unsigned  i = 0; i < EXTENT; i++) {
		_func[i] = resolve_or_throw(lib, FUNC_NAMES[i]);
	}
}

//---------------------------------------------------------------------------
// Life Cycle
Kernel::Kernel(std::string const &kernel_lib) : _kernel_lib(kernel_lib), _xsi(_kernel_lib) {}

Kernel::~Kernel() {
	if(_design_lib)  std::cerr << "Disposing XSI Kernel with open Design." << std::endl;
}

void Kernel::open(std::string const &design_lib, s_xsi_setup_info const &setup_info) {
	_design_lib.open(design_lib);
	try {
		auto      const  f   = t_fp_xsi_open(resolve_or_throw(_design_lib, "xsi_open"));
		xsiHandle const  hdl = f(const_cast<p_xsi_setup_info>(&setup_info));
		if(!hdl)  throw  std::runtime_error("Loading of design failed");
		_xsi.setHandle(hdl);

		// Enumerate Ports
		unsigned const          port_count = xsi<Xsi::get_int>(xsiNumTopPorts);
		std::unique_ptr<Port[]> ports { new Port[port_count] };
		for(unsigned  i = 0; i < port_count; i++)  new(&ports[i]) Port(*this, i);
		_port_count = port_count;
		_ports = std::move(ports);
	}
	catch(...) {
		_design_lib.close();
		throw;
	}
}
void Kernel::close() noexcept {
	xsi<Xsi::close>();
	_xsi.setHandle(nullptr);
	_design_lib.close();
	_ports.reset();

	// Clean up Library State
	std::optional<void*> const  vptr = _kernel_lib.getsymbol("svTypeInfo");
	if(vptr) *((void**)*vptr) = nullptr;
}

//===========================================================================
// xsi::Port

bool Port::hasUnknown() const {
	unsigned                   const  n = (width()+31) / 32;
	s_xsi_vlog_logicval const *const  p = buf();
	for(unsigned  i = 0; i < n; i++) {
		if(p[i].bVal)  return  true;
	}
	return  false;
}

bool Port::isZero() const {
	unsigned                   const  n = (width()+31) / 32;
	s_xsi_vlog_logicval const *const  p = buf();
	for(unsigned  i = 0; i < n; i++) {
		if(p[i].aVal)  return  false;
	}
	return  true;
}

std::string Port::as_binstr() const {
	unsigned const  w = width();
	std::string  res(w, '?');

	s_xsi_vlog_logicval const *si = buf();
	std::string::iterator      di = res.end();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
	uint32_t  a;
	uint32_t  b;
	for(unsigned  i = 0; i < w; i++) {
		if((i & 31) == 0) {
			a = si->aVal;
			b = si->bVal;
			si++;
		}
		*--di = XZ10[((b&1)<<1)|(a&1)];
		a >>= 1;
		b >>= 1;
	}
#pragma GCC diagnostic pop
	return  res;
}

std::string Port::as_hexstr() const {
	unsigned  l = (width()+3)/4;
	std::string  res(l, '?');
	s_xsi_vlog_logicval const *si = buf();
	std::string::iterator      di = res.end();

	while(l > 0) {
		uint32_t  a = si->aVal;
		uint32_t  b = si->bVal;
		si++;

		unsigned  m = std::min(8u, l);
		l -= m;
		do {
			unsigned const  bm = b & 0xF;
			unsigned const  am = a & 0xF;

			*--di = !bm? HEX[am] : XZ10[3 - !(am&bm)];
			a >>= 4;
			b >>= 4;
		}
		while(--m > 0);
	}
	return  res;
}

Port& Port::clear() {
	unsigned             const  n = (width()+31) / 32;
	s_xsi_vlog_logicval *const  p = buf();
	std::fill(p, p+n, s_xsi_vlog_logicval { .aVal = 0u, .bVal = 0u });
	return *this;
}

Port& Port::set_binstr(std::string const &val) {
	std::string::const_iterator  si = val.end();
	s_xsi_vlog_logicval         *di = buf();

	unsigned const  n = (width()+31) / 32;
	unsigned  l = val.length();
	for(unsigned  i = 0; i < n; i++) {
		uint32_t  a = 0;
		uint32_t  b = 0;

		unsigned const  m = std::min(32u, l);
		l  -= m;
		si -= m;
		for(unsigned  j = 0; j < m; j++) {
			a <<= 1;
			b <<= 1;
			switch(*si++) {
			case '1':
				a |= 1;
			case '0':
				continue;

			default:
				a |= 1;
			case 'Z':
			case 'z':
				b |= 1;
				continue;
			}
		}
		si -= m;

		di->aVal = a;
		di->bVal = b;
		di++;
	}

	return *this;
}

Port& Port::set_hexstr(std::string const &val) {
	std::string::const_iterator  si = val.end();
	s_xsi_vlog_logicval         *di = buf();

	unsigned const  n = (width()+31) / 32;
	unsigned  l = val.length();
	for(unsigned  i = 0; i < n; i++) {
		uint32_t  a = 0;
		uint32_t  b = 0;

		unsigned const  m = std::min(8u, l);
		l  -= m;
		si -= m;
		for(unsigned  j = 0; j < m; j++) {
			char  c = *si++;
			a <<= 4;
			b <<= 4;

			if(('0' <= c) && c <= '9')  a |= c & 0xF;
			else {
				c |= 0x20;
				if(('a' <= c) && (c <= 'f'))  a |= c - ('a'-10);
				else {
					b |= 0xF;
					if(c != 'z')  a |= 0xF;
				}
			}
		}
		si -= m;

		di->aVal = a;
		di->bVal = b;
		di++;
	}

	return *this;
}
