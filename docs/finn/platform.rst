.. _platform:

***********************************
Targets, parts, and platform data
***********************************

A FINN build targets a **part** (``fpga_part``, for example ``"xczu7ev-ffvc1156-2-e"``)
or a **board** (``board``, for example ``"ZCU104"``):

* A part alone is sufficient for stitched IP, RTL simulation, and out-of-context synthesis
  builds.
* Shell integration (Zynq and Versal PYNQ, SLASH) and bitfile generation require a
  compatible board.

Both are defined in :py:mod:`finn.platform`, which reads them from **platform data**:

* For a part: its device, the architecture and DSP block of that device, whether it is a
  Versal, Zynq-7000, or Zynq UltraScale+ part, and its resources in total and per SLR.
* For a board: its part, its shell flow, its Vivado board preset, its default Vitis
  platform, its memories and memory ports, and the earliest Vivado that builds for it.

FINN ships platform data for all 7-series, Zynq-7000, UltraScale, UltraScale+, and Versal
parts in the part database of its supported Vivado version. You can update this registry for
a newer Vivado, a new architecture, or a custom device or board, as described in the
sections below.

Use the ``finn-platform`` command to see what FINN resolves for a part or a board, and
where the data comes from::

    finn-platform show xczu7ev-ffvc1156-2-e
    finn-platform show ZCU104
    finn-platform sources


Using parts from a newer Vivado
===============================

Put the newer Vivado on your ``PATH`` (``source settings64.sh``). Then extract the parts
that FINN does not hold yet, for example::

    finn-platform extract --parts 'xc2ve3858*' --out ~/fpga/vivado-new

The extraction reads from the part database of Vivado and never synthesizes. It has two passes:

* A *registry pass* lists the matching parts with their devices and resources. It takes
  seconds for a family.
* A *link pass* counts the resources of each SLR. For each device with several SLRs, it
  links an empty design on one part of that device. This takes about a minute per device.
  Devices with one SLR need no link.

``--jobs N`` runs N Vivado processes at once. ``--work DIR`` keeps the Vivado logs in
``DIR``. The command checks the files that it writes. It leaves out the parts of an
architecture that no platform data defines, and reports them.

Then list the directory in ``FINN_PLATFORM_PATH``::

    export FINN_PLATFORM_PATH=~/fpga/vivado-new

``run-docker.sh`` forwards ``FINN_PLATFORM_PATH`` into FINN's container and mounts each
directory that it lists.


Using a new architecture
========================

FINN's built-in platform data defines every architecture of its supported Vivado version. A
newer Vivado may add an architecture that no platform data defines. The extraction leaves
out its parts and names the architecture. Define it in an ``architectures.json`` in the
output directory (see `Defining a custom device`_), then extract again.


Defining a custom device
========================

Custom devices can be added by hand. Each file is optional:

``manifest.json``
    The name of the data and where it comes from::

        {"name": "my-devices", "source": {"kind": "hand-written", "note": "made up"}}

``devices.jsonl``
    One device per line. Each line gives the ``ARCHITECTURE`` and ``FAMILY`` that Vivado
    states for the parts of the device, and the key of its resource record::

        {"name": "xcmy100t", "architecture": "virtex7", "family": "virtex7", "resources": "my100t"}

``resources.jsonl``
    Resource records, one per line. Each record has a unique key (``digest``), the totals
    and the counts of each SLR. The counts are CLB LUTs, CLB registers, RAMB18 blocks
    (two per RAMB36), URAM288 blocks and DSP slices::

        {"digest": "my100t", "slr_count": 1,
         "totals": {"lut": 100000, "ff": 200000, "bram18": 250, "uram": 0, "dsp": 350},
         "slrs": [{"lut": 100000, "ff": 200000, "bram18": 250, "uram": 0, "dsp": 350}]}

    In the file, each record is on one line. It is wrapped here. The SLRs must sum to the
    totals.

``architectures.json``
    For an architecture that no platform data defines yet: its DSP block, its
    processing-system class (``"versal"``, ``"zynq7000"``, ``"zynquplus"`` or ``null``)
    and whether it has UltraRAM::

        {"myarch": {"dsp": "DSP48E1", "class": null, "uram": false}}

A part belongs to the device whose name is the longest prefix of the part name. The
comparison ignores case. For example, ``xcmy100tffg1761-2`` is on ``xcmy100t``. Check the
files before you use them::

    finn-platform check ~/fpga/my-devices

FINN checks platform data each time it loads it. It reports a problem with the directory,
file, line and rule, for example
``'my-devices' (...), resources.jsonl:1: slr-sum: the SLRs sum to ..., not the totals ...``.

Then put the files in a directory, and list it in ``FINN_PLATFORM_PATH``.


Adding a board
==============

A board is a row in a ``boards.json``. The key of the row is the name that a build gives as
``board``::

    {
      "MyZU7Board": {
        "part": "xczu7ev-ffvc1156-2-e",
        "shell": "VIVADO_ZYNQ",
        "board_part": "vendor.com:myzu7board:part0:1.0",
        "axi_port_width": 128,
        "memory": ["DDR"]
      }
    }

The fields, where they apply:

* ``part``: the FPGA part of the board (required).
* ``shell``: the shell flow that FINN builds bitfiles with (``VIVADO_ZYNQ``,
  ``VIVADO_VERSAL``, ``VITIS_ALVEO`` or ``SLASH_ALVEO``, which builds for the V80's device
  only). Without it, a build for the board stops before a bitfile.
* ``board_part``: the Vivado board preset of the Zynq and Versal shells, as the
  ``board.xml`` of the board states it (``vendor:name:part0:file_version``). Vivado must
  find the board files. FINN adds ``$FINN_ROOT/deps/board_files`` to the board repository
  paths of Vivado, and Vivado finds its own bundled boards.
* ``axi_port_width``: the width in bits of the memory port that the DMAs of the Zynq and
  Versal shells use (128 on the Zynq UltraScale+ boards).
* ``vitis_platform``: the default Vitis platform of an Alveo card.
* ``memory``: the memories of the board, preferred first (``DDR``, ``HBM``).
* ``mem_port``: the memory port that the DMAs of an Alveo card connect to by default, for
  example ``"HBM[0]"``, or ``"DDR[{slr}]"`` for the DDR bank of the SLR of each DMA. Its
  memory is one of ``memory``.
* ``vivado_min``: the earliest Vivado release that builds for the board, for example
  ``"2024.1"``.
* ``golden``: the Versal shell builds on a golden reference design for the board.
* ``ps_config``: Zynq UltraScale+ PS settings that the Zynq shell applies after the preset,
  for example ``{"PSU__USE__M_AXI_GP0": "1"}``.
* ``retired``: why the board is no longer supported. FINN refuses a build that names the
  board.
* ``note``: free text for readers.

``finn-platform check DIR`` also compares the ``part`` and ``board_part`` of each row with
the ``board.xml`` of the board, where the board files are present. It looks in
``$FINN_ROOT/deps/board_files`` and in the board store bundled with Vivado, which it finds
through ``XILINX_VIVADO``.

To ship a board with FINN, add its row to ``src/finn/platform/data/boards.json``.

Otherwise, put the ``boards.json`` in a directory, and list it in ``FINN_PLATFORM_PATH``.


Building for a retired board
============================

The row of a retired board stays in FINN's built-in platform data, marked ``retired``.
Copy its row from FINN's built-in ``boards.json`` (``src/finn/platform/data/boards.json``
in a checkout, ``finn/platform/data/boards.json`` in an installed package) into a
``boards.json`` of your own, under a name of your own and without ``retired``. Name that
board in your build. ``finn-platform show Pynq-Z2`` prints the facts FINN resolves for the
board, but its output is not a row to copy.

Then put the ``boards.json`` in a directory, and list it in ``FINN_PLATFORM_PATH``.


Refusals
========

The configuration check of the build reports each refusal as an error, named by its code:

=================================  ==================================================================
Code                               When
=================================  ==================================================================
``unknown-board``                  the board is in no platform data
``retired-board``                  the board is retired (the message says why)
``board-part-mismatch``            ``board`` and an ``fpga_part`` that is not the board's part
``target-unstated``                neither ``board`` nor ``fpga_part``, where a part is needed
``unknown-part``                   no platform data holds the device of the part
``board-shell-mismatch``           ``shell_flow_type`` is not the board's shell flow
``shell-architecture-mismatch``    a shell flow for a part of an architecture it does not build for
``shell-device-mismatch``          a shell flow for a device it does not build for (SLASH: V80 only)
``no-shell``                       a bitfile or driver, but the board or the build has no shell flow
``board-required``                 a bitfile with the Zynq, Versal or SLASH shell flow, but no board
``no-board-preset``                a bitfile for a board whose row has no ``board_part``
``vitis-platform-required``        a Vitis bitfile with no ``vitis_platform`` and no default
``vitis-platform-unused``          a ``vitis_platform`` given for another shell flow, or with none
``vivado-too-old``                 Vivado is older than the board's or the shell flow's minimum
``vivado-unknown``                 a minimum applies, but ``XILINX_VIVADO`` names no Vivado release
``golden-missing``                 the Versal shell's golden reference files are missing
``memory-unstated``                a DMA or loop needs a memory port or type, and no board states it
=================================  ==================================================================


Regenerating FINN's built-in platform data
==========================================

Maintainers regenerate the built-in devices from a new Vivado release with the same
command. It writes into ``src/finn/platform/data`` and keeps ``architectures.json`` and
``boards.json`` there, which are written by hand::

    finn-platform extract --parts '*' --out src/finn/platform/data --name built-in --jobs 4 --work /tmp/extract

To make the same platform data again from the logs in ``--work``, without Vivado, use
``--from-logs /tmp/extract``.
