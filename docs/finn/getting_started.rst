.. _getting_started:

***************
Getting Started
***************

Quickstart
==========

1. Install Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
2. Set up ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` environment variables pointing respectively to the Xilinx tools installation directory and version (e.g. ``FINN_XILINX_PATH=/opt/Xilinx`` and ``FINN_XILINX_VERSION=2024.2``)
3. Clone the FINN compiler from the repo: ``git clone https://github.com/Xilinx/finn/`` and go into the directory where it is cloned
4. Execute ``./docker/run --fpga -- bash docker/quicktest.sh verify`` to verify your installation. It is normal to see warnings during the tests - FINN uses warnings to inform users about certain conditions. As long as all tests pass, your installation is successful.
5. Optionally, follow the instructions on :ref:`getting_started:PYNQ board first-time setup`, :ref:`getting_started:Vitis-based Alveo first-time setup`, or :ref:`getting_started:Slash-based Alveo first-time setup` for board setup.
6. Optionally, set up a `Vivado/Vitis license`_.
7. All done! See :ref:`getting_started:Running FINN in Docker` for the various options on how to run the FINN compiler.


How do I use FINN?
==================

We strongly recommend that you first watch one of the pre-recorded `FINN tutorial <https://www.youtube.com/watch?v=zw2aG4PhzmA&amp%3Bindex=2>`_
videos, then follow the Jupyter notebook tutorials for `training and deploying an MLP for network intrusion detection <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/cybersecurity>`_ .
You may also want to check out the other :ref:`tutorials`, and the `FINN examples repository <https://github.com/Xilinx/finn-examples>`_ .

Our aim in FINN is *not* to accelerate common off-the-shelf neural networks, but instead provide you with a set of tools
to train *customized* networks and create highly-efficient FPGA implementations from them.
In general, the approach for using the FINN framework is as follows:

1. Train your own quantized neural network (QNN) in `Brevitas <https://github.com/Xilinx/brevitas>`_. We have some `guidelines <https://bit.ly/finn-hls4ml-qat-guidelines>`_ on quantization-aware training (QAT).
2. Export to QONNX and convert to FINN-ONNX by following `this tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb>`_ .
3. Use FINN's ``build_dataflow`` system on the exported model by following this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb>`_ or for advanced settings have a look at this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb>`_ .
4. Adjust your QNN topology, quantization settings and ``build_dataflow`` configuration to get the desired results.

Please note that the framework is still under development, and how well this works will depend on how similar your custom network is to the examples we provide.
If there are substantial differences, the default build flow may need additional transformations for your network,
or new functions (including Vitis HLS layers) may be required.
In that case, keep using the builder and add the transformations your network needs as custom build steps,
which you can insert at any point of the default flow with ``inject_steps_before`` and ``inject_steps_after``,
as shown in the `advanced builder settings tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb>`_.
To find out where the flow needs to change, inspect the intermediate models that the builder saves after every step
and enable its verification steps.
The `BNN-PYNQ notebooks <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/bnn-pynq>`_ explain what each builder step does
and how to simulate the intermediate models, and the other `advanced FINN tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/advanced>`_
show how to write your own transformations and custom operations.
Once you have a working flow, you can run it from the command line
by using the "advanced mode" described in the :ref:`command_line` section.

Running FINN in Docker
======================
FINN uses Ubuntu 24.04, Python 3.12 and a uv-locked environment. The
``docker/build`` command prepares it; ``docker/run`` reuses it and installs
the mounted FINN checkout editable without resolving dependencies.

Launch interactive shell
************************
Start a shell; configured Vivado/Vitis installations and licences are mounted
automatically::

  ./docker/run

To require the tools (failing early if none are configured), or to leave them out::

  ./docker/run --fpga
  ./docker/run --no-tools

Launch a Build with ``build_dataflow``
**************************************
For an external model directory, mount it explicitly. Paths are mirrored inside
the container, including paths with spaces::

  ./docker/run --fpga --volume "/path/to/model:/path/to/model" -- build_dataflow /path/to/model

For a custom Python flow, also select its working directory::

  ./docker/run --fpga --volume "/path/to/model:/path/to/model" --workdir /path/to/model -- python build.py

Vitis/Alveo builds additionally select the XRT userspace runtime with
``--runtime xrt``. Jenkins selects that full profile by default.

Launch Jupyter notebooks
************************
Serve the upstream notebooks, with token authentication enabled by default::

  ./docker/run --fpga --notebook

Open the token URL printed by Jupyter. ``JUPYTER_PORT`` defaults to 8888 and
``NETRON_PORT`` to 8081. Set ``JUPYTER_PASSWD_HASH`` to use a hashed password.

Environment variables
**********************
The host resolver reads ``~/.config/finn/xilinx.env``. Environment variables take
precedence; ``FINN_XILINX_ENV`` selects another file (empty disables it).
Write plain ``NAME=value`` lines, without quotes or expansion::

  FINN_XILINX_PATH=/opt/Xilinx
  FINN_XILINX_VERSION=2024.2
  XILINXD_LICENSE_FILE=2100@licence-server

Both AMD installation layouts are detected. Software runs work without tools.

* ``PLATFORM_REPO_PATHS`` names colon-separated platform directories (default ``/opt/xilinx/platforms`` where it exists).
* ``FINN_TOOL_DIR_OVERRIDE`` retains site tool interception, including LSF wrappers.
* ``FINN_HOST_BUILD_DIR`` selects the disposable build directory (default ``/tmp/finn_build_<uid>``).
* ``FINN_HOME`` selects persistent state such as resources and XSI builds (default ``~/.finn``); ``FINN_RESOURCES_DIR``, ``FINN_XSI_PATH`` and ``FINN_GOLDEN_DIR`` move parts of it elsewhere.
* ``NUM_DEFAULT_WORKERS`` and ``FINN_XELAB_MT`` retain their existing meanings.
* ``FINN_DOCKER_EXTRA`` passes shell-quoted Docker flags; they are tokenized without shell evaluation.
* ``FINN_DOCKER_CACHE_DIR`` retains Torch and Hugging Face download caches.
* ``FINN_SSH_KEY_DIR`` (default: the checkout's ``ssh_keys`` directory, where it exists) is mounted read-only as the container's ``~/.ssh``.
* ``FINN_VCK190_GOLDEN=1`` builds the optional VCK190 golden reference into ``FINN_HOME``.

See ``docker/README.md`` for the full interface migration, resource overrides,
runtime packages, shared images, and Apptainer/Singularity exports.

General FINN Docker tips
************************
* Several folders including the root directory of the FINN compiler and the ``FINN_HOST_BUILD_DIR`` will be mounted into the Docker container and can be used to exchange files.
* Do not use ``sudo`` to launch the FINN Docker. Instead, setup Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
* If you want a new terminal on an already-running container, you can do this with ``docker exec -it <name_of_container> bash``.
* The container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the finn compiler folder (which is mounted from the host computer) or otherwise backed up.

Running FINN without Docker (Local Installation)
=================================================

For environments where Docker is not available, FINN can be installed and run locally.
Note that Docker remains the primary supported method.

Prerequisites
*************

* Ubuntu 24.04 (other distributions may work but are not officially tested)
* Python 3.12
* System dependencies (see below)
* Vivado/Vitis 2024.2 or later (for synthesis and simulation)

Quick Start
***********

1. Install system dependencies (requires sudo)::

    sudo ./scripts/install-system-deps.sh

2. Set up Xilinx tools environment variables::

    export FINN_XILINX_PATH=/opt/Xilinx
    export FINN_XILINX_VERSION=2024.2

3. Install uv 0.10.0, then clone FINN and run the local setup script::

    git clone https://github.com/Xilinx/finn.git
    cd finn
    ./setup-local.sh

   If your system Python is not 3.12, set ``FINN_PYTHON`` to point to a Python 3.12 interpreter::

    export FINN_PYTHON=/path/to/python3.12
    ./setup-local.sh

4. Activate the FINN environment::

    source scripts/activate.sh

5. Verify the installation::

    ./scripts/quicktest-local.sh

Setup Script Options
********************

The ``setup-local.sh`` script supports several options:

* ``--help``: Show usage information
* ``--ci``: CI mode (non-interactive, fail fast on errors)
* ``--skip-xsi``: Skip building finn_xsi (Vivado Python interface)
* ``--skip-resources``: Use prepared resource caches or local overrides. Python dependencies still follow the lock.

Validation Test Modes
*********************

The ``quicktest-local.sh`` script supports different test modes:

* ``./scripts/quicktest-local.sh``: Run basic tests (imports, transformations, utilities)
* ``./scripts/quicktest-local.sh vivado``: Also run a sanity test to check if Vivado integration works (cppsim, rtlsim)

If you plan to use Vivado for synthesis and simulation, we recommend running
``./scripts/quicktest-local.sh vivado`` to verify that the Vivado integration is
working correctly.

Limitations
***********

The local installation has some limitations compared to Docker:

* System dependency versions may vary from the tested Docker environment
* XRT (Xilinx Runtime) must be installed separately for Alveo support
* Some edge cases may behave differently due to environment differences

If you encounter issues, please try the Docker-based installation first to verify the
issue is not environment-specific.

Supported FPGA Hardware
=======================
**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx-AMD FPGA as part of a larger system. It’s up to you to take the FINN-generated accelerator (what we call “stitched IP” in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <http://www.pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the AUP-ZU3, Kria SOM, Ultra96, ZCU102 and ZCU104 boards, as well as UltraScale+-based Alveo datacenter accelerator cards.

Retired boards (Pynq-Z1/Pynq-Z2)
********************************
The Zynq-7000 based Pynq-Z1 and Pynq-Z2 boards were retired from official support with the move to Vivado 2024.2. The AUP-ZU3 (a Zynq UltraScale+ board from the AMD University Program) is the recommended supported replacement for academic use.

These boards are no longer officially supported and have been removed from our CI and testing, so we make no guarantees about them. That said, the build flow itself is unchanged and Vivado 2024.2 still supports the ``xc7z020`` part, so you can re-enable them at your own decision by removing them from the ``retired_pynq_boards`` set in ``src/finn/util/basic.py``. The board entries in ``pynq_part_map``, ``pynq_native_port_width`` and ``util/platforms.py`` are kept in place. Supply the retired board files in a local assembled tree selected by ``FINN_BOARD_FILES_PATH``, or add a pinned ``vivado-boards`` resource declaration to your project's ``pyproject.toml``.

PYNQ board first-time setup
****************************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PYNQ board. To be able to access the target from the host, you'll need to set up SSH public key authentication:

Start on the target side:

1. Note down the IP address of your PYNQ board. This IP address must be accessible from the host.
2. Ensure the ``bitstring`` package is installed: ``sudo pip3 install bitstring``

Continue on the host side (replace the ``<PYNQ_IP>`` and ``<PYNQ_USERNAME>`` with the IP address and username of your board from the first step):

1. Create the ``ssh_keys`` directory where you cloned finn (``mkdir ssh_keys``), then launch the Docker container from there with ``./docker/run --fpga``
2. Go into the `ssh_keys` directory  (e.g. ``cd /path/to/finn/ssh_keys``)
3. Run ``ssh-keygen`` to create a key pair e.g. ``id_rsa`` private and ``id_rsa.pub`` public key
4. Run ``ssh-copy-id -i id_rsa.pub <PYNQ_USERNAME>@<PYNQ_IP>`` to install the keys on the remote system
5. Test that you can ``ssh <PYNQ_USERNAME>@<PYNQ_IP>`` without having to enter the password. Pass the ``-v`` flag to the ssh command if it doesn't work to help you debug.


Vitis-based Alveo first-time setup
**********************************
The Vitis toolchain targets UltraScale and UltraScale+-based Alveo cards, such as the U250. We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.



On the host side:

1. Install Vitis 2024.2 and set up the ``VITIS_PATH`` environment variable to point to your installation.
2. Install Xilinx XRT on the host if you want to run accelerators there; the FINN image gets its pinned XRT userspace with ``./docker/build --runtime xrt``.
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.
5. Done!

Slash-based Alveo first-time setup
***********************************
The Slash toolchain targets Versal-based Alveo cards such as the V80 using the ``slashkit``
linker. We use *host* to refer to the PC running the FINN Docker environment, which will
build the accelerator and package it up, and *target* to refer to the PC where the V80
card is installed. These two can be the same PC, or connected over the network.

Prior to first usage, you need to build the Slash packages from source and set up both
the host and the target. Please refer to the `Slash GitHub repository
<https://github.com/Xilinx/slash>`_ for instructions on how to build all Slash packages,
including the ``slashkit`` linker package.

On the target side:

1. Install all Slash runtime packages as described in the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_.
2. Done!

On the host side:

1. Build the ``slashkit`` Debian package from the `Slash GitHub repository
   <https://github.com/Xilinx/slash>`_ and copy it to a location accessible on the host.
2. Supply the packages named by ``docker/runtimes/slash.env`` and
   ``docker/runtimes/slashkit.env`` under ``docker/packages/``. Select them with
   ``./docker/build --runtime xrt --runtime slash --runtime slashkit`` and use
   the same runtime options when launching.
3. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_.
   Copy your private key to the ``finn/ssh_keys`` folder on the host to get
   password-less deployment and remote execution.
4. Done!

Vivado/Vitis license
*********************
If you are targeting Xilinx FPGA parts that needs specific licenses (non-WebPack) you can make these available to the
FINN Docker container by passing extra arguments. To do this, you can use the ``FINN_DOCKER_EXTRA`` environment variable as follows:

::

  export FINN_DOCKER_EXTRA=" -v /path/to/licenses:/path/to/licenses -e XILINXD_LICENSE_FILE=/path/to/licenses "

The above example mounts ``/path/to/licenses`` from the host into the same path on the Docker container, and sets the
value of the ``XILINXD_LICENSE_FILE`` environment variable.

System Requirements
====================

* A Linux host with ``bash`` (the FINN Docker image provides the required OS environment, so the host distribution/version is not critical)
* Docker `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* A working Vitis/Vivado 2024.2 installation
* ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` environment variables correctly set, see `Quickstart`_
* *(optional)* `Vivado/Vitis license`_ if targeting non-WebPack FPGA parts.
* *(optional)* A PYNQ board with a network connection, see `PYNQ board first-time setup`_

We also recommend running the FINN compiler on a system with sufficiently
strong hardware:

* **RAM.** Depending on your target FPGA platform, your system must have sufficient RAM to be
  able to run Vivado/Vitis synthesis for that part. See `this page <https://www.xilinx.com/products/design-tools/vivado/vivado-ml.html#memory>`_
  for more information. For targeting Zynq and Zynq UltraScale+ parts, at least 8 GB is recommended. Larger parts may require up to 16 GB.
  For targeting Alveo parts with Vitis or Slash, at least 64 GB RAM is recommended.

* **CPU.** FINN can parallelize HLS synthesis and several other operations for different
  layers, so using a multi-core CPU is recommended. However, this should be balanced
  against the memory usage as a high degree of parallelization will require more
  memory. See the ``NUM_DEFAULT_WORKERS`` environment variable below for more on
  how to control the degree of parallelization.

* **Storage.** While going through the build steps, FINN will generate many files as part of
  the process. For larger networks, you may need 10s of GB of space for the temporary
  files generated during the build.
  By default, these generated files will be placed under ``/tmp/finn_build_<uid>``.
  You can override this location by using the ``FINN_HOST_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.
