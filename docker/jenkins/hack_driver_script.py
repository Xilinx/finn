import os

def remove_cache_dirs(dir_list):
    tmp_list = list(dir_list)
    for i in range(len(tmp_list)-1, -1, -1):
        if ".pytest_cache" in tmp_list[i]:
            del tmp_list[i]
        elif "__pycache__" in tmp_list[i]:
            del tmp_list[i]
    return tmp_list

def hack_driver_script(board, test_dir):
    test_script_file = "driver.py"
    # Read the contents of the test script file
    with open(test_script_file, "r") as f:
        lines = f.readlines()

    # Specify the line to be replaced and the new line
    line_to_replace = "ishape_normal"
    if "cnv" in test_dir:
        new_line = "    \"ishape_normal\" : [(1, 3, 32, 32)],"
    else:
        new_line = "    \"ishape_normal\" : [(1, 1, 28, 28)],"

    # Iterate over the lines and replace the specified line
    for i in range(len(lines)):
        if line_to_replace in lines[i]:
            lines[i] = new_line + "\n"
            break  # Only replace the first occurrence

    # Write the modified contents back to the test script file
    with open(test_script_file, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    current_dir = os.getcwd()
    board = os.path.basename(current_dir)

    # Get list of local directories - removing the Python cache directories
    local_dirs = [name for name in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, name))]
    local_dirs = remove_cache_dirs(local_dirs)

    # Now create the full paths for each relative path
    local_dirs_full_path = [os.path.join(current_dir, name) for name in local_dirs if os.path.isdir(os.path.join(current_dir, name))]

    # Change the driver.py script for each of the test directories
    for dir in local_dirs_full_path:
        os.chdir(dir)
        hack_driver_script(board, dir)
