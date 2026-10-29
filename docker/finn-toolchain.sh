# shellcheck shell=bash
# Shared vendor settings for native activation, entrypoint and direct-exec shims.
# Resolve paths on the host with launcher.py; source settings only with Bash.

# Fail LOUDLY rather than silently doing nothing. Sourcing this from dash used
# to leave PATH untouched and produce a 127 several steps later, which points
# nowhere near the cause. stderr is fine here; only stdout must stay clean.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "finn: /etc/finn-toolchain.sh needs bash (settings64.sh uses \`source\`);" >&2
    echo "finn: this shell is not bash, so the Xilinx toolchain was NOT applied." >&2

# Already applied by a parent shell, an outer source, or a shim. Nested shells
# are extremely common (make, pytest, Vivado's own Tcl shelling out).
elif [ "${FINN_ENV_APPLIED:-}" != "1" ]; then

    # ---------------------------------------------------------------------
    # 1. Source every settings script that is present, in order.
    #
    # Vitis, Vivado, HLS, XRT. This is the order the previous resolver used,
    # and each script PREPENDS to PATH, so the order decides which tool wins
    # when two releases ship the same binary name. Preserved deliberately: it
    # is what FINN's flows were validated against. Do not reorder to taste.
    #
    # `|| true` on each: settings64.sh reads unset variables on several Xilinx
    # releases. A shell that aborts partway leaves a half-configured PATH which
    # fails much later and somewhere else.
    # ---------------------------------------------------------------------
    _finn_nounset=0
    case $- in *u*) _finn_nounset=1; set +u ;; esac
    for _finn_base in \
        "${XILINX_VITIS:-}" "${XILINX_VIVADO:-}" "${XILINX_HLS:-}" \
        "${XILINX_XRT:-}" /opt/xilinx/xrt
    do
        [ -n "$_finn_base" ] || continue
        for _finn_name in settings64.sh setup.sh; do
            _finn_script="$_finn_base/$_finn_name"
            case " ${_finn_sourced:-} " in *" $_finn_script "*) continue ;; esac
            if [ -f "$_finn_script" ]; then
                # shellcheck disable=SC1090
                . "$_finn_script" >/dev/null 2>&1 || true
                _finn_sourced="${_finn_sourced:-} $_finn_script"
            fi
        done
    done

    # ---------------------------------------------------------------------
    # 2. Library paths settings64.sh does NOT provide.
    #
    #   lib/lnx64.o             the Vivado simulation kernel, which finn_xsi
    #                           loads
    #   lnx64/tools/fpo_v7_1    the floating-point operator libraries that
    #                           HLS-generated code links against
    #
    # Direct `docker exec` sessions also need these settings.
    # ---------------------------------------------------------------------
    for _finn_dir in \
        "${XILINX_VIVADO:-}/lib/lnx64.o" \
        "${XILINX_VITIS:-}/lnx64/tools/fpo_v7_1" \
        "${XILINX_HLS:-}/lnx64/tools/fpo_v7_1"
    do
        case "$_finn_dir" in /lib/lnx64.o|/lnx64/tools/fpo_v7_1) continue ;; esac
        [ -d "$_finn_dir" ] || continue
        LD_LIBRARY_PATH="$_finn_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    [ -n "${LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH

    # ---------------------------------------------------------------------
    # 3. The FLEXlm/libudev workaround, for hosts that have no baked ENV.
    #
    # Without it a licence checkout dies with "realloc(): invalid pointer"
    # inside udev_enumerate_scan_devices: FLEXlm fingerprints the machine
    # through libudev during checkout and corrupts the heap doing it.
    #
    # The IMAGE bakes this as ENV (docker/Dockerfile.finn), asserted at build
    # time, so the branch below is a no-op there. The bare-host lane has no
    # image to bake into, and had its own copy of this in TWO files --
    # scripts/activate.sh and setup-local.sh, byte-identical. Both now get it
    # from here, which is the file whose stated job is applying the toolchain
    # wherever it runs.
    # ---------------------------------------------------------------------
    case ":${LD_PRELOAD:-}:" in
        *libudev.so.1*) ;;
        *)
            # -H: /lib is a symlink to usr/lib on merged-/usr hosts.
            _finn_libudev=$(find -H /lib -maxdepth 2 -path '*-linux-gnu/libudev.so.1' -print -quit 2>/dev/null)
            if [ -n "$_finn_libudev" ]; then
                LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$_finn_libudev"
                export LD_PRELOAD
            fi
            ;;
    esac

    # ---------------------------------------------------------------------
    # 4. De-duplicate the path variables.
    #
    # settings64.sh PREPENDS unconditionally, so applying the toolchain more
    # than once in a process tree grows PATH without bound. Observed at four
    # copies of the full Xilinx PATH, about 3 kB -- slow to search and enough
    # to make any diagnostic that prints the environment unreadable.
    #
    # Order-preserving, first occurrence wins, so the toolchain still shadows
    # the system tools exactly as settings64.sh intended.
    # ---------------------------------------------------------------------
    # Farm interception belongs to resolve_xilinx_tool(), not bare/nested calls.
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        PATH="$VIRTUAL_ENV/bin:$PATH"
        export PATH
    fi

    for _finn_var in PATH LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH; do
        eval "_finn_val=\${$_finn_var:-}"
        [ -n "$_finn_val" ] || continue
        _finn_out=""
        _finn_ifs=$IFS; IFS=:
        for _finn_part in $_finn_val; do
            [ -n "$_finn_part" ] || continue
            case ":$_finn_out:" in *":$_finn_part:"*) continue ;; esac
            _finn_out="${_finn_out:+$_finn_out:}$_finn_part"
        done
        IFS=$_finn_ifs
        eval "export $_finn_var=\$_finn_out"
    done

    [ "$_finn_nounset" != 1 ] || set -u
    unset _finn_nounset
    FINN_ENV_APPLIED=1
    export FINN_ENV_APPLIED

    unset _finn_base _finn_name _finn_script _finn_sourced _finn_dir \
          _finn_var _finn_val _finn_out _finn_ifs _finn_part _finn_libudev
fi

# Never let this file's last command decide the caller's exit status.
true
