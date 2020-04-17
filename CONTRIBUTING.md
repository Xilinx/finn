We welcome contributions to FINN. Please first sign our CLA.

Please follow the steps below and be sure that your contribution complies with our guidelines.

1. Share your proposal via <a href="https://github.com/Xilinx/finn/issues" target="_blank">Github issues</a>.

	We welcome submissions to:

		1. The FINN flow like additional custom ONNX nodes, transformation and analysis passes.
		2. Contributions to the documentation and Jupyter notebooks

	If you want to add example networks, we ask you to make them into a separate repo.

2. Submitting your pull request:

	1. Fork this repository to your own github account using the *fork* button above.

	2. Clone the fork to a local computer using *git clone*. Checkout the branch you want to workon.

	3. You can modify the Python source code, Jupyter notebooks and Sphinx documentation

	4. Please install <a href="https://pre-commit.com/" target="_blank">pre-commit</a>. The hooks we use for pre-commit can be found in <a href="https://github.com/Xilinx/finn/blob/master/.pre-commit-config.yaml" target="_blank">this file</a>

	5. Use *git add*, *git commit*, *git push* to add changes to your fork.

	6. Submit a pull request by clicking the *pull request* button on your github repo:
		1. The <a href="https://github.com/Xilinx/finn" target="_blank">master branch</a> should always be treated as stable and clean. Only hot fixes are allowed to be pull-requested. The hot fix is supposed to be very important such that without this fix, a lot of things break.
        2. For new features, small bug fixes, doc updates, and many other fixes, users should pull request against the <a href="https://github.com/Xilinx/finn/tree/dev" target="_blank">development branch</a>.

3. We will review your contribution and, if any additional fixes or modifications are
necessary, may provide feedback to guide you. When accepted, your pull request will
be merged to the repository. For questions please contact us via <a href="https://gitter.im/xilinx-finn/community" target="_blank">gitter channel</a>.
