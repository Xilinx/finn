.. _getting_started:

***************
Getting Started
***************

.. note:: **This website is currently under construction.**

How to use the FINN compiler
============================
The FINN compiler should not be thought of a single pushbutton tool that does everything for you, but rather as a collection of scripts/tools that will help you convert a QNN into a custom FPGA accelerator that performs high-performance inference. We do provide several examples of taking trained networks all the way down to FPGA bitfiles, but if you are trying to do this for custom networks you will have to write your own Python scripts that call the appropriate FINN Compiler functions that process your design correctly, or adding new functions as required.

Requirements
============

* Ubuntu 18.04 with ``bash`` installed
* Docker
* A working Vivado 2019.1 installation
* A ``VIVADO_PATH`` environment variable pointing to the Vivado installation directory (e.g. the directory where settings64.sh is located)
* (optional) A PYNQ board with a network connection
   * the ``bitstring`` package must be installed on the PYNQ: ``sudo pip3 install bitstring``
* (optional) An Alveo board, and a working Vitis 2020.1 installation if you want to use Vitis and Alveo (see `Alveo first-time setup`_ below)


Running FINN in Docker
======================
We use Docker extensively for developing and deploying FINN. If you are not familiar with Docker, there are many excellent `online resources <https://docker-curriculum.com/>`_ to get started. There is a Dockerfile in the root of the repository, as well as a `run-docker.sh` script that can be launched in the following modes:

Getting an interactive shell for development or experimentation
***************************************************************
.. note:: **run-docker.sh requires bash to execute correctly.**

::

  ./run_docker.sh

Simply running sh run-docker.sh without any additional arguments will clone the dependency repos, create a Docker container and give you a terminal with you can use for development for experimentation.
If you want a new terminal on an already-running container, you can do this with `docker exec -it finn_dev_<username> bash`.

.. warning:: The Docker container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the /workspace/finn folder (which is mounted from the host computer) or otherwise backed up.

.. note:: **Develop from host, run inside container:** The FINN repository directory will be mounted from the host, so that you can use a text editor on your host computer to develop and the changes will be reflected directly inside the container.

Running the Jupyter notebooks
*****************************
::

  ./run-docker.sh notebook

This will launch the `Jupyter notebook <https://jupyter.org/>`_ server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones.
.. note:: The link will look something like this (the token you get will be different):
http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc

The run-docker.sh script forwards ports 8888 for Jupyter and 8081 for Netron, and launches the notebook server with appropriate arguments.

Running the test suite directly
*******************************
FINN comes with a set of tests to check for regressions. The full test suite
(which will take several hours to run and require a PYNQ board) can be executed
by:

::

  ./run-docker.sh test

There is a quicker variant of the test suite that skips the tests marked as
requiring Vivado or as slow-running tests:

::

  ./run-docker.sh quicktest

If you want to run individual tests, you can do this *inside the Docker container
from the FINN root directory* as follows:

::

  python setup.py test --addopts "-k test_end2end_tfc_w1a2"

Finally, if you want to run tests in parallel (e.g. to take advantage of a multi-core CPU)
you can use:
 * pytest-parallel for any rtlsim tests, e.g. `python setup.py test --addopts "-k rtlsim --workers auto"`
 * pytest-xdist for anything else, make sure to add `--dist=loadfile` if you have tests in the same file that have dependencies on each other e.g. `python setup.py test --addopts "-k mytest -n auto --dist=loadfile"`

Please see the pytest documentation for more about picking tests by marks or by name.

Environment variables
**********************

Prior to running the `run-docker.sh` script, there are several environment variables you can set to configure certain aspects of FINN.
These are summarized below:

* ``VIVADO_PATH`` points to your Vivado installation on the host
* (optional, for Vitis & Alveo only) ``VITIS_PATH``, ``PLATFORM_REPO_PATHS`` and ``XILINX_XRT`` respectively point to your Vitis installation, the Vitis platform files, and Xilinx XRT
* ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* ``NUM_DEFAULT_WORKERS`` (default 1) specifies the degree of parallelization for the transformations that can be run in parallel
* ``PYNQ_BOARD`` or ``ALVEO_BOARD`` specifies the type of PYNQ/Alveo board used (see "supported hardware" below) for the test suite
* ``PYNQ_IP`` and ``PYNQ_PORT`` (or ``ALVEO_IP`` and ``ALVEO_PORT``) specify ip address and port number to access the PYNQ board / Alveo target
* ``PYNQ_USERNAME`` and ``PYNQ_PASSWORD`` (or ``ALVEO_USERNAME`` and ``ALVEO_PASSWORD``) specify the PYNQ board / Alveo host access credentials for the test suite. For PYNQ, password is always needed to run as sudo. For Alveo, you can leave the password empty and place your ssh private key in the ``finn/ssh_keys`` folder to use keypair authentication.
* ``PYNQ_TARGET_DIR`` (or ``ALVEO_TARGET_DIR``) specifies the target dir on the PYNQ board / Alveo host for the test suite

Supported Hardware
===================
**End-to-end support including driver:** For quick deployment, FINN targets boards supported by  `PYNQ <https://pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the Pynq-Z1, Pynq-Z2, Ultra96, ZCU102 and ZCU104 boards. 
As of FINN v0.4b we also have preliminary support for `Xilinx Alveo boards <>`_ using PYNQ and Vitis, see instructions below for Alveo setup.

**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx FPGA as part of a larger system. It's up to you to take the FINN-generated accelerator (what we call "stitched IP" in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

Alveo first-time setup
**********************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT and set up the ``XILINX_XRT`` environment variable to point to your installation.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.
6 (optional) If you don't want to specify the ``ALVEO_PASSWORD`` environment variable, you can `set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.


On the host side:

1. Install Vitis 2020.1 and set up the ``VITIS_PATH`` environment variable to point to your installation. 
2. Install Xilinx XRT and set up the ``XILINX_XRT`` environment variable to point to your installation. *This must be the same path as the target's XRT (target step 1)*
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
4. Set up the ``ALVEO_*`` environment variables accordingly for your target, see description of environment variables above.
5. Done! You can try the ``test_end2end_vitis`` tests in the FINN Docker to verify your setup, although this will take some time.
