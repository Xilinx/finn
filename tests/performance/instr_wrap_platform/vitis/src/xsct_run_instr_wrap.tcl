set PLATFORM_NAME [lindex $argv 0]
set PLATFORM_XSA ${PLATFORM_NAME}_wrapper.xsa

set CLK_FREQ [lindex $argv 1]

set HW_SERVER_HOST [lindex $argv 2]
set HW_SERVER_PORT [lindex $argv 3]


# If workspace already exists, delete it and create a new empty one
file delete -force -- ./test_workspace
file mkdir ./test_workspace

# Set the workspace
setws ./test_workspace

file delete -force ../src/instr_wrap.cpp

# Set the clock frequency in the instrumentation wrapper source file
set instr_wrap_temp [open "../src/instr_wrap_template.cpp" r]
set instr_wrap_file [open "../src/instr_wrap.cpp" w]
while {[gets $instr_wrap_temp line] >= 0} {
    set newline [string map [subst {CLOCK_FREQUENCY $CLK_FREQ}] $line]
    puts $instr_wrap_file $newline
}
close $instr_wrap_temp
close $instr_wrap_file

# Create and build the platform using the exported .xsa
platform create -name "test_platform" -hw $PLATFORM_XSA -proc psv_cortexa72_0 -os standalone
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
targets 6

# Reset the processor
rst -processor -clear-registers -skip-activate-subsystem

# Download the instrumentation wrapper elf file onto the hardware
dow ./test_workspace/instr_wrap/Debug/instr_wrap.elf

# Run the instrumentation wrapper
con

# Disconnect from the server
disconnect
