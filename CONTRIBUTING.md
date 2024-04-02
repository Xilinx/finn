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

3. Sign Your Work

Please use the *Signed-off-by* line at the end of your patch which indicates that you accept the Developer Certificate of Origin (DCO) defined by https://developercertificate.org/ reproduced below::

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
```

You can enable Signed-off-by automatically by adding the `-s` flag to the `git commit` command.

Here is an example Signed-off-by line which indicates that the contributor accepts DCO:

```
  This is my commit message

  Signed-off-by: Jane Doe <jane.doe@example.com>
```

4. We will review your contribution and, if any additional fixes or modifications are
necessary, may provide feedback to guide you. When accepted, your pull request will
be merged to the repository. If you have more questions please contact us.
