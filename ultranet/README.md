## UltraNet on FINN


This is a flow for building UltraNet models based on the repo here [https://github.com/heheda365/ultra_net](https://github.com/heheda365/ultra_net)

### Setup
1. Clone FINN and checkout the appropriate branch (add the recursive flag to get the ultra_net submodule)
```bash
git clone org-3189299@github.com:Xilinx/finn.git -b feature/ultranet --recursive
```

2. Enter the docker container, **ensure your env is setup appropriately [details here](https://finn.readthedocs.io/en/latest/getting_started.html#quickstart)**
```bash
cd finn
./run-docker.sh
```
3. Inside the docker container navigate to the ultranet folder and make
```bash
cd ultranet
make all
```

The run will take quite a while, it will produce the FINN artifacts in `finn/ultranet/output_ultranet_fpga`
