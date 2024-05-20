set PLATFORM_NAME [lindex $argv 0]
set PLATFORM_XSA ${PLATFORM_NAME}_wrapper.xsa

set HW_SERVER_HOST [lindex $argv 1]
set HW_SERVER_PORT [lindex $argv 2]


# If workspace already exists, delete it and create a new empty one
file delete -force -- ./test_workspace
file mkdir ./test_workspace

# Set the workspace
setws ./test_workspace

# Create and build the platform using the exported .xsa
platform create -name "test_platform" -hw $PLATFORM_XSA -proc CIPS_0_pspmc_0_psv_cortexa72_0 -os standalone
platform active test_platform
platform generate

# Create and build the instrumentation wrapper application
app create -name instr_wrap -platform test_platform -template {Empty Application (C++)} -lang c++
importsources -name instr_wrap -path ../src/instr_wrap.cpp -target-path src
app build -name instr_wrap

# Connect to the remote hardware server
connect -host $HW_SERVER_HOST -port $HW_SERVER_PORT

# Select the processor used to build the platform
# (Cortex-A72)
targets -set -nocase -filter {name =~ "*A72*#0"}

# Reset the processor
rst -processor -clear-registers -skip-activate-subsystem

# Download the instrumentation wrapper elf file onto the hardware
dow ./test_workspace/instr_wrap/Debug/instr_wrap.elf

# Run the instrumentation wrapper
con
puts "The instrumentation wrapper is currently running..."
after 10000
puts "The instrumentation wrapper has finished running"

# Disconnect from the server
disconnect
