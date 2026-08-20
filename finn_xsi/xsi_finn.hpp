/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FINN XSI++: C++ XSI Binding used by FINN.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/
#ifndef XSI_FINN_HPP
#define XSI_FINN_HPP

#include <memory>
#include <optional>
#include <tuple>
#include <string>
#include <cstring>

#include <stdexcept>
#include <exception>

#if defined(_WIN32)
#	include <windows.h>
#else
#	include <dlfcn.h>
#endif

#include "xsi.h"


namespace xsi {

//===========================================================================
// Shared Library Representation

class SharedLibrary {
public:
	static char const  library_suffix[];

private:
	using  handle_type =
#if defined(_WIN32)
		HINSTANCE;
#else
		void*;
#endif

	//-----------------------------------------------------------------------
	// Instance State
private:
	handle_type _lib;
	std::string _path;

	//-----------------------------------------------------------------------
	// Life Cycle
public:
	SharedLibrary() : _lib(nullptr), _path() {}
	SharedLibrary(std::string const &path) : _lib(load(path)), _path(path) {}
	~SharedLibrary() { unload(); }

private:
	SharedLibrary(SharedLibrary const&) = delete;
	SharedLibrary& operator=(SharedLibrary const&) = delete;

public:
	operator bool() const { return  bool(_lib); }
	SharedLibrary& open(std::string const &path);
	SharedLibrary& close() {
		unload();
		_lib = nullptr;
		_path.clear();
		return *this;
	}

private:
	static handle_type load(std::string const &path);
	void unload();

	//-----------------------------------------------------------------------
	// Accessors
public:
	std::string const& path() const { return _path; }
	std::optional<void*> getsymbol(char const *const  name);

}; // class SharedLibrary

//===========================================================================
// xsi::Kernel

template<typename It>
class enumerator {
	It _begin;
	It _end;
public:
	enumerator(It  begin, It  end) : _begin(begin), _end(end) {}
	~enumerator() {}
public:
	It begin() const { return _begin; }
	It end()   const { return _end; }
};

class Design;
class Port;
class Kernel {

	//-----------------------------------------------------------------------
	// Dispatch Table for XSI Functions
	class Xsi {
		//- Statics ---------------------
	public:
		// Function Indeces
		static constexpr unsigned
			get_value = 0, put_value = 1,
			get_int_port = 2, get_str_port = 3,

			get_int = 4, get_port_number = 5,

			trace_all = 6, run = 7, restart = 8,
			get_status = 9, get_error_info = 10,

			close = 11;

	private:
		// Function Names & Types
		static constexpr unsigned  EXTENT = 12;
		static char const *const  FUNC_NAMES[EXTENT];
		using  type_map = std::tuple<
			// Port Access
			t_fp_xsi_get_value, t_fp_xsi_put_value,
			t_fp_xsi_get_int_port, t_fp_xsi_get_str_port,

			// Design Inspection
			t_fp_xsi_get_int, t_fp_xsi_get_port_number,

			// Simulation Control & Status
			t_fp_xsi_trace_all, t_fp_xsi_run, t_fp_xsi_restart,
			t_fp_xsi_get_status, t_fp_xsi_get_error_info,

			// Closing
			t_fp_xsi_close
		>;

		//- Actual Contents -------------
	private:
		xsiHandle _hdl;
		void*     _func[EXTENT];

		//- Lifecycle: in-place structure inside Kernel only
	public:
		Xsi(SharedLibrary &lib);
		~Xsi() {}
	private:
		Xsi(Xsi const&) = delete;
		Xsi& operator=(Xsi const&) = delete;

		//- Handle Update ---------------
	public:
		void setHandle(xsiHandle  hdl) { _hdl = hdl; }

		//- XSI Function Invocation -----
	public:
		template<unsigned  FID, typename... Args>
		auto invoke(Args&&... args) const {
			auto const  f = decltype(std::get<FID>(type_map()))(_func[FID]);
			return  (*f)(_hdl, std::forward<Args>(args)...);
		}

	}; // class Xsi

private:
	// Instance State
	SharedLibrary _kernel_lib;	// Backing Kernel Library
	Xsi           _xsi;       	// XSI Dispatch Table

	// Optional State once a Design in open
	SharedLibrary           _design_lib;
	unsigned                _port_count;
	std::unique_ptr<Port[]> _ports;

public:
	Kernel(std::string const &kernel_lib);
	Kernel(Kernel const&) = delete;
	Kernel& operator=(Kernel const&) = delete;
	~Kernel();

	// Interface reserved for forwarded access through open Design
private:
	friend Design;
	friend Port;
	template<unsigned  FID, typename... Args>
	auto xsi(Args&&... args) const {
		return _xsi.invoke<FID>(std::forward<Args>(args)...);
	}

	// Port Accessors inlined below and public through Design
	Port*       getPort(char const *const  name);
	Port const* getPort(char const *const  name) const;
	enumerator<Port*>       ports();
	enumerator<Port const*> ports() const;

	// Design con- & destruction hooks
	void open(std::string const &design_lib, s_xsi_setup_info const &setup_info);
	void close() noexcept;

public:
	// Hex printing manipulation
	static void hex_in_lower();
	static void hex_in_upper();

}; // class Kernel

//===========================================================================
// xsi::Design

//	- non-copyable, non-movable handle for exposing simulation control.
class Design {
	using  Xsi = Kernel::Xsi;
	Kernel &_kernel;

public:
	Design(
		Kernel &kernel,
		std::string const &design_lib,
		s_xsi_setup_info const &setup_info
	) : _kernel(kernel) { kernel.open(design_lib, setup_info); }
	Design(
		Kernel &kernel, std::string const &design_lib,
		char const *const  log_file = nullptr,
		char const *const  wdb_file = nullptr
	) : Design(kernel, design_lib, s_xsi_setup_info {
		.logFileName = const_cast<char*>(log_file),
		.wdbFileName = const_cast<char*>(wdb_file)
	}) {}
	~Design() { _kernel.close(); }

private:
	Design(Design const&) = delete;
	Design& operator*(Design const&) = delete;

	//-----------------------------------------------------------------------
	// Forwarded Access to Open Simulation

	// Simulation Control & Status
public:
	void trace_all()                { _kernel.xsi<Xsi::trace_all>(); }
	void run(XSI_INT64 const  step) { _kernel.xsi<Xsi::run>(step); }
	void restart()                  { _kernel.xsi<Xsi::restart>(); }

	int         get_status()     const { return _kernel.xsi<Xsi::get_status>(); }
	char const* get_error_info() const { return _kernel.xsi<Xsi::get_error_info>(); }

	// Port Access
public:
	int num_ports() const { return _kernel._port_count; }

	Port*       getPort(std::string const &name)       { return _kernel.getPort(name.c_str()); }
	Port const* getPort(std::string const &name) const { return _kernel.getPort(name.c_str()); }

	enumerator<Port*>       ports()       { return _kernel.ports(); }
	enumerator<Port const*> ports() const { return const_cast<Kernel const&>(_kernel).ports(); }

}; // class Design

//===========================================================================
// xsi::Port

// Only exists within controlled environment within Kernel with open Design.
class Port {
	using  Xsi = Kernel::Xsi;
	Kernel         &_kernel;
	unsigned const  _id;
	std::unique_ptr<s_xsi_vlog_logicval[]> const _buf;

private:
	// Con- and destruction under full control of Kernel
	friend class Kernel;
	Port() : _kernel(*static_cast<Kernel*>(nullptr)), _id(0), _buf() {}
	Port(Kernel &kernel, unsigned const  id)
		: _kernel(kernel), _id(id),
		_buf(std::make_unique<s_xsi_vlog_logicval[]>((width()+31)/32)) {}
	Port(Port const&) = delete;
	Port& operator=(Port const&) = delete;
public:
	~Port() {}

public:
	char const* name()  const { return _kernel.xsi<Xsi::get_str_port>(_id, xsiNameTopPort); }
	int         dir()   const { return _kernel.xsi<Xsi::get_int_port>(_id, xsiDirectionTopPort); }
	unsigned    width() const { return _kernel.xsi<Xsi::get_int_port>(_id, xsiHDLValueSize); }

	bool isInput()  const { return  dir() == xsiInputPort; }
	bool isOutput() const { return  dir() == xsiOutputPort; }
	bool isInout()  const { return  dir() == xsiInoutPort; }

private:
	s_xsi_vlog_logicval*       buf()       { return _buf.get(); }
	s_xsi_vlog_logicval const* buf() const { return _buf.get(); }

public:
	// Buffer Synchronization
	Port& read() {
		_kernel.xsi<Xsi::get_value>(_id, buf());
		return *this;
	}
	void write_back() {
		_kernel.xsi<Xsi::put_value>(_id, buf());
	}

	// Inspection
	bool hasUnknown() const;
	bool isZero() const;
	bool operator[](unsigned const  idx) const {
		return (buf()[idx/32].aVal >> (idx%32)) & 1;
	}

	bool     as_bool()     const { return  buf()->aVal & 1; }
	unsigned as_unsigned() const { return  buf()->aVal; }
	std::string as_binstr() const;
	std::string as_hexstr() const;

	// Manipulation
	Port& clear();
	Port& set(unsigned  val) {
		s_xsi_vlog_logicval *const  p = buf();
		p->aVal = val;
		p->bVal =   0;
		return *this;
	}
	Port& set_binstr(std::string const &val);
	Port& set_hexstr(std::string const &val);

}; // class Port

// Inlined Kernel Port Accessors

inline Port* Kernel::getPort(char const *const  name) {
	int const  id = xsi<Xsi::get_port_number>(name);
	return  (id == -1)? nullptr : &_ports[id];
}
inline Port const* Kernel::getPort(char const *const  name) const {
	int const  id = xsi<Xsi::get_port_number>(name);
	return  (id == -1)? nullptr : &_ports[id];
}

inline enumerator<Port*> Kernel::ports() {
	Port *const  beg = _ports.get();
	return { beg, beg + _port_count };
}
inline enumerator<Port const*> Kernel::ports() const {
	Port const *const  beg = _ports.get();
	return { beg, beg + _port_count };
}

} // namespace xsi

#endif
