We welcome contributions to FINN.

Please follow the steps below and be sure that your contribution complies with our guidelines.

1. Share your proposal via <a href="https://github.com/Xilinx/finn/issues" target="_blank">Github issues</a>. If you are looking for some issues to get started with, we have a list of <a href="https://github.com/Xilinx/finn/labels/good%20first%20issue">good first issues</a> in the issue tracker. Feel free to ask questions in the <a href="https://github.com/Xilinx/finn/discussions">FINN GitHub discussions</a> as well.

	We welcome submissions to:

	1. The FINN flow like additional custom ONNX nodes, transformation and analysis passes.
	2. Contributions to the documentation and Jupyter notebooks

	To ensure clean separation of toolflow and examples, we do not keep example networks in this repo. If you want to add example networks, we ask you to make them into a separate repo and use FINN as a dependency -- we'll be happy to add it to the list of <a href="https://xilinx.github.io/finn/community">FINN community projects</a>.

2. Submitting your pull request:

	1. Fork this repository to your own GitHub account using the *fork* button above.

	2. Clone the fork to your local computer using *git clone*. Checkout the branch you want to work on.

	3. Please install <a href="https://pre-commit.com/" target="_blank">pre-commit</a> to ensure your code is formatted to our style guidelines. The hooks we use for pre-commit can be found in <a href="https://github.com/Xilinx/finn/blob/main/.pre-commit-config.yaml" target="_blank">this file</a>

	4. Modify the Python source code, Jupyter notebooks and Sphinx documentation etc. as needed.

	5. Use *git add*, *git commit*, *git push* to add changes to your fork.

	6. If you are introducing new functionality, add at least one unit test under the `test/` folder and make sure it passes before you submit the pull request.

	7. Submit a pull request by clicking the *pull request* button on your GitHub repo:
		1. The <a href="https://github.com/Xilinx/finn" target="_blank">main branch</a> should always be treated as stable and clean. Only hot fixes are allowed to be pull-requested. The hot fix is supposed to be very important such that without this fix, a lot of things will break.
        2. For new features, smaller bug fixes, doc updates, and many other fixes, users should pull request against the <a href="https://github.com/Xilinx/finn/tree/dev" target="_blank">development branch</a>.

3. We will review your contribution and, if any additional fixes or modifications are
necessary, may provide feedback to guide you. When accepted, your pull request will
be merged to the repository. If you have more questions please contact us.
