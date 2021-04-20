.. _getting_started:

***************
Getting Started
***************

How to use the FINN compiler
============================
Currently, it's best to think of the FINN compiler as *compiler infrastructure*
instead of a full *compiler* like `gcc` (although the aim is to get there).
Although we provide a :ref:`command_line` entry for building dataflow
accelerators, this only exposes a basic flow that works for simpler networks.
A better way of looking at the FINN compiler is as a collection of scripts/tools that will help
you convert a QNN into a custom FPGA accelerator that performs high-performance inference.

**So where do I get started?** The best way of getting started with the FINN
compiler is to follow the existing
`Jupyter notebooks <tutorials>`_ and check out the prebuilt
`examples <https://github.com/Xilinx/finn-examples>`_.

**How do I compile my custom network?**
This depends on how similar your custom network is to the examples we provide.
If there are substantial differences, you will most likely have to write your own
Python scripts that call the appropriate FINN compiler
functions that process your design correctly, or adding new functions (including
Vivado HLS layers)
as required.
For custom networks, we recommend making a copy of the end-to-end
Jupyter notebook as a starting point, visualizing the model at intermediate
steps and adding calls to new transformations as needed.
Once you have a working flow, you can implement a command line entry for this
by using the "advanced mode" described in the :ref:`command_line` section.




System Requirements
====================

* Ubuntu 18.04 with ``bash`` installed
* Docker `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* A working Vivado 2019.1 or 2020.1 installation
* A ``VIVADO_PATH`` environment variable pointing to the Vivado installation directory (e.g. the directory where settings64.sh is located)
* *(optional)* A PYNQ board with a network connection
   * the ``bitstring`` package must be installed on the PYNQ: ``sudo pip3 install bitstring``
* *(optional)* An Alveo board, and a working Vitis 2020.1 installation if you want to use Vitis and Alveo (see `Alveo first-time setup`_ below)

We also recommend running the FINN compiler on a system with sufficiently
strong hardware:

* **RAM.** Depending on your target FPGA platform, your system must have sufficient RAM to be
  able to run Vivado/Vitis synthesis for that part. See `this page <https://www.xilinx.com/products/design-tools/vivado/memory.html>`_
  for more information. For targeting Zynq and Zynq UltraScale+ parts, at least 8 GB is recommended. Larger parts may require up to 16 GB.
  For targeting Alveo parts with Vitis, at least 64 GB RAM is recommended.

* **CPU.** FINN can parallelize HLS synthesis and several other operations for different
  layers, so using a multi-core CPU is recommended. However, this should be balanced
  against the memory usage as a high degree of parallelization will require more
  memory. See the ``NUM_DEFAULT_WORKERS`` environment variable below for more on
  how to control the degree of parallelization.

* **Storage.** While going through the build steps, FINN will generate many files as part of
  the process. For larger networks, you may need 10s of GB of space for the temporary
  files generated during the build.
  By default, these generated files will be placed under ``/tmp/finn_dev_<username>``.
  You can override this location by using the ``FINN_HOST_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.


Running FINN in Docker
======================
We use Docker extensively for developing and deploying FINN. If you are not familiar with Docker, there are many excellent `online resources <https://docker-curriculum.com/>`_ to get started. There is a Dockerfile in the root of the repository, as well as a `run-docker.sh` script that can be launched in the following modes:

Getting an interactive shell for development or experimentation
***************************************************************
.. warning:: Do not use ``sudo`` to launch the FINN Docker. Instead, setup Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_

::

  bash ./run_docker.sh

Simply running sh run-docker.sh without any additional arguments will clone the dependency repos, create a Docker container and give you a terminal with you can use for development for experimentation.
If you want a new terminal on an already-running container, you can do this with `docker exec -it finn_dev_<username> bash`.

.. warning:: The Docker container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the /workspace/finn folder (which is mounted from the host computer) or otherwise backed up.

.. note:: **Develop from host, run inside container:** The FINN repository directory will be mounted from the host, so that you can use a text editor on your host computer to develop and the changes will be reflected directly inside the container.

Command Line Entry
*******************
FINN is currently more compiler infrastructure than compiler, but we do offer
a :ref:`command_line` entry for certain use-cases. These run a predefined flow
or a user-defined flow from the command line as follows:

::

  bash ./run_docker.sh build_dataflow <path/to/dataflow_build_dir/>
  bash ./run_docker.sh build_custom <path/to/custom_build_dir/>


Running the Jupyter notebooks
*****************************
::

  bash ./run-docker.sh notebook

This will launch the `Jupyter notebook <https://jupyter.org/>`_ server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones.
.. note:: The link will look something like this (the token you get will be different):
http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc

The ``run-docker.sh`` script forwards ports 8888 for Jupyter and 8081 for Netron, and launches the notebook server with appropriate arguments.


Environment variables
**********************

Prior to running the `run-docker.sh` script, there are several environment variables you can set to configure certain aspects of FINN.
These are summarized below:

* ``VIVADO_PATH`` points to your Vivado installation on the host
* (optional, for Vitis & Alveo only) ``VITIS_PATH``, ``PLATFORM_REPO_PATHS`` and ``XILINX_XRT`` respectively point to your Vitis installation, the Vitis platform files, and Xilinx XRT
* (optional) ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* (optional) ``JUPYTER_PASSWD_HASH`` (default "") Set the Jupyter notebook password hash. If set to empty string, token authentication will be used (token printed in terminal on launch).
* (optional) ``LOCALHOST_URL`` (default localhost) sets the base URL for accessing e.g. Netron from inside the container. Useful when running FINN remotely.
* (optional) ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* (optional) ``NUM_DEFAULT_WORKERS`` (default 1) specifies the degree of parallelization for the transformations that can be run in parallel
* (optional) ``PYNQ_BOARD`` or ``ALVEO_BOARD`` specifies the type of PYNQ/Alveo board used (see "supported hardware" below) for the test suite
* (optional) ``PYNQ_IP`` and ``PYNQ_PORT`` (or ``ALVEO_IP`` and ``ALVEO_PORT``) specify ip address and port number to access the PYNQ board / Alveo target
* (optional) ``PYNQ_USERNAME`` and ``PYNQ_PASSWORD`` (or ``ALVEO_USERNAME`` and ``ALVEO_PASSWORD``) specify the PYNQ board / Alveo host access credentials for the test suite. For PYNQ, password is always needed to run as sudo. For Alveo, you can leave the password empty and place your ssh private key in the ``finn/ssh_keys`` folder to use keypair authentication.
* (optional) ``PYNQ_TARGET_DIR`` (or ``ALVEO_TARGET_DIR``) specifies the target dir on the PYNQ board / Alveo host for the test suite
* (optional) ``FINN_HOST_BUILD_DIR`` specifies which directory on the host will be used as the build directory. Defaults to ``/tmp/finn_dev_<username>``
* (optional) ``IMAGENET_VAL_PATH`` specifies the path to the ImageNet validation directory for tests.

Supported Hardware
===================
**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <https://pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the Pynq-Z1, Pynq-Z2, Ultra96, ZCU102 and ZCU104 boards.
As of FINN v0.4b we also have preliminary support for `Xilinx Alveo boards <https://www.xilinx.com/products/boards-and-kits/alveo.html>`_ using PYNQ and Vitis, see instructions below for Alveo setup.

**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx FPGA as part of a larger system. It's up to you to take the FINN-generated accelerator (what we call "stitched IP" in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

Zynq first-time setup
**********************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PYNQ board. To be able to access the target from the host, you'll need to set up SSH public key authentication:

Start on the target side:

* Note down the IP address of your PYNQ board. This IP address must be accessible from the host.

Continue on the host side (replace the <PYNQ_IP> and <PYNQ_USERNAME> with the IP address and username of your board from the first step):

* Launch the Docker container from where you cloned finn with ``./run-docker.sh``
* Go into the `ssh_keys` directory  (e.g. ``cd /workspace/finn/ssh_keys``)
* Run ``ssh-keygen`` to create a key pair e.g. ``id_rsa`` private and ``id_rsa.pub`` public key
* Run ``ssh-copy-id -i id_rsa.pub <PYNQ_USERNAME>@<PYNQ_IP>`` to install the keys on the remote system
* Test that you can ``ssh <PYNQ_USERNAME>@<PYNQ_IP>`` without having to enter the password. Pass the ``-v`` flag to the ssh command if it doesn't work to help you debug.


Alveo first-time setup
**********************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT and set up the ``XILINX_XRT`` environment variable to point to your installation, for instance ``/opt/xilinx/xrt``.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.
6. (optional) If you don't want to specify the ``ALVEO_PASSWORD`` environment variable, you can `set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.


On the host side:

1. Install Vitis 2020.1 and set up the ``VITIS_PATH`` environment variable to point to your installation.
2. Install Xilinx XRT and set up the ``XILINX_XRT`` environment variable to point to your installation. *This must be the same path as the target's XRT (target step 1)*
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. Set up the ``ALVEO_*`` environment variables accordingly for your target, see description of environment variables above.
5. Done! You can try the ``test_end2end_vitis`` tests in the FINN Docker to verify your setup, although this will take some time.
