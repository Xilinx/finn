## FINN Instrumentation Wrapper Jupyter Notebooks

### **Requires 2022.2 Tools**

A pair of notebooks that go through a simple example of building and running the instrumentation wrapper to capture the performance of the TFC-w1a1 model built using the FINN compiler. The first notebook (`1-build_model_and_platform.ipynb`) details building the model using the FINN compiler with additional steps to produce `.xo` kernel files, then using these files to build the hardware platform. The second notebook (`2-run_instr_wrap.ipynb`) details the running of the instrumentation wrapper on the target board. \
\
**Note: Due to a bug with XSCT using Docker, the instrumentation wrapper is unable to run from the notebook (`2-run_instr_wrap.ipynb`). It must be run from outside of the notebook. The notebook will detail the process through which the instrumentation wrapper is run.**
