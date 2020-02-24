***************
Getting Started
***************

How to use the FINN compiler
============================
The FINN compiler should not be thought of a single pushbutton tool that does everything for you, but rather as a collection of scripts/tools that will help you convert a QNN into a custom FPGA accelerator that performs high-performance inference. We do provide several examples of taking trained networks all the way down to FPGA bitfiles, but if you are trying to do this for custom networks you will have to write your own Python scripts that call the appropriate FINN Compiler functions that process your design correctly, or adding new functions as required.

Requirements
============

* Ubuntu 18.04
* Docker
* A working Vivado installation
* A `VIVADO_PATH` environment variable pointing to the Vivado installation directory (e.g. the directory where settings64.sh is located)

Running FINN in Docker
======================
We use Docker extensively for developing and deploying FINN. If you are not familiar with Docker, there are many excellent `online resources <https://docker-curriculum.com/>`_ to get started. There is a Dockerfile in the root of the repository, as well as a `run-docker.sh` script that can be launched in the following modes:

Getting an interactive shell for development or experimentation
***************************************************************
::

  sh run_docker.sh
   
Simply running sh run-docker.sh without any additional arguments will clone the dependency repos, create a Docker container and give you a terminal with you can use for development for experimentation.

.. warning:: The Docker container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the /workspace/finn folder (which is mounted from the host computer) or otherwise backed up.

.. note:: **Develop from host, run inside container:** The FINN repository directory will be mounted from the host, so that you can use a text editor on your host computer to develop and the changes will be reflected directly inside the container.

Running the Jupyter notebooks
*****************************
::

  sh run-docker.sh notebook

This will launch the Jupyter notebook server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones. 
.. note:: The link will look something like this (the token you get will be different):
http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc

The run-docker.sh script forwards ports 8888 for Jupyter and 8081 for Netron, and launches the notebook server with appropriate arguments.

Running the test suite directly
*******************************
::
  
  sh run-docker.sh test

FINN comes with a set of tests which can be launched using the command above. Note that some of the tests involve extra compilation and the entire test suite may take some time to complete.  

Running the test suite using Jenkins
************************************
::

  sh run-docker.sh jenkins

