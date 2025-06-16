from .kernel_registry import KernelRegistry, gkr
from .kernel import Kernel, KernelInvalidParameter, KernelProjection

import importlib
import os

# Get the current package's directory  
package_dir = os.path.dirname(__file__)  

# Iterate over all items in the current directory  
for item in os.listdir(package_dir):  
    item_path = os.path.join(package_dir, item)  

    # Check if the item is a directory and contains an __init__.py  
    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):  
        # Import the subpackage to run its __init__.py  
        importlib.import_module(f'.{item}', package=__name__)
