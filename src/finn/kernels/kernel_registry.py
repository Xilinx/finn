from dataclasses import dataclass, field
from .kernel import Kernel, KernelInvalidParameter
from typing import Dict, List, Callable, Optional
from .cache_manager import cache_manager


class KernelRegistryOverwriteDefault(Exception):
    def __init__(self, message: str):  
        super().__init__(message)  
        self.message = message  

    def __str__(self):  
        return f"Attempted to overwrite default kernel: {self.message}"  

@dataclass
class KernelBucket:

    _bucket : Dict[int, list[Kernel]] = field(default_factory=dict)

    """ Allows inserting kernels based on priority, but also allows handling of special cases with extra logic. """
    def _put(self, k: Kernel, priority: int) -> None:

        if k in self._priority_list():
            return

        if priority not in self._bucket: # If priority level not seen before, instantiate KernelBucket.
            self._bucket[priority] = [k]
            return

        if priority == 0: # Trying to add another default kernel, raise error.
            raise KernelRegistryOverwriteDefault(f"Tried to overwrite default kernel {self._bucket[0]} with {k}")

        self._bucket[priority].append(k) # Add kernel to bucket.

    """ Return all kernels in bucket, sorted on priority first, reverse insertion order second. """
    def _priority_list(self) -> list[Kernel]:
        # Sort buckets based on priority first, reverse insertion order second.
        sorted_buckets = [list(reversed(self._bucket[i])) for i in sorted(self._bucket.keys())]
        # Flatten buckets (lists) of kernels into just kernels.
        return [k for bucket in sorted_buckets for k in bucket]


class KernelRegistryMeta(type):
    """ Meta class for ensuring the KernelRegistry is a Singleton. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(KernelRegistryMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class KernelRegistry(metaclass=KernelRegistryMeta):
    """
    This is a class for the kernel registry.

    It is a singleton and ensures that it is created on module instantiation.
    Any subsequent calls to new to create a new registry will be a reference to the original object.
    It uses the KernelRegistryMeta class to ensure this behaviour.

    Care should be taken when using this singleton with multiprocessing.
    """    

    def __init__(self):
        self._mapping : Dict[str, KernelBucket] = {}

    def __deepcopy__(self, memo):
        # Return the same object to ensure this is a singleton.
        return self

    def kernel_exists(self, op_type: str) -> bool:
        """ Check if a customop has any kernels registered. No constraints checking. """
        return self._mapping.get(op_type) != None

    def register(self, op_type: str, k: Kernel, priority: int = -2) -> None:
        """ Register a kernel at a particular priority. """  
        if not (op_type in self._mapping.keys()):
            # First time this customop is seen, make a new bucket for it.
            self._mapping[op_type] = KernelBucket()

        # Append kernel with priority. p=0 is default, p<0 is high priority, p>0 is low priority.
        self._mapping[op_type]._put(k, priority)

    def kernel(self, op_type: str, config: dict, cost_fn: Callable[[List[Kernel]],Kernel] = None) -> Kernel:
        """
        For a given set of attributes find a suitable kernel that satistfies the 
        attributes according to the kernel priority list and user defined cost function.

        All viable candidates are found in priority order, the best fit is chosen by cost_fn. If no viable candidate
        was found then None is returned.
        """

        candidates : list[Kernel] = self._mapping[op_type]._priority_list()
        candidates_viable: List[Kernel] = []
        for candidate in candidates:
            try:
                candidates_viable.append(candidate(**config))
            except KernelInvalidParameter:
                print(f"Not possible to instantiate the kernel {candidate}, trying the next one for {config=}")
                print(f"{candidates=}")
                pass
        
        if cost_fn == None:
            def default_cost_fn(candidates_viable: List[Kernel]):
                if len(candidates_viable) == 0:
                    return None
                else:
                    return candidates_viable[0]
            cost_fn = default_cost_fn

        return cost_fn(candidates_viable)

    def clear_cache(self, op_type: Optional[str] = None, kernel_class: Optional[str] = None) -> int:
        """Clear cached files for kernels.
        
        Args:
            op_type: Operation type to clear cache for (if None, clears all)
            kernel_class: Kernel class to clear cache for (if None, clears all)
            
        Returns:
            Number of cache entries invalidated
        """
        return cache_manager.invalidate_cache(op_type, kernel_class)

    def cleanup_cache(self) -> int:
        """Remove expired cache entries from disk.
        
        Returns:
            Number of entries cleaned up
        """
        return cache_manager.cleanup_expired()

    def cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return cache_manager.get_cache_stats()

    def check_shared_files_changed(self, op_type: str) -> List[str]:
        """Check if shared files have changed for kernels of a given operation type.
        
        Args:
            op_type: Operation type to check
            
        Returns:
            List of changed file paths
        """
        if op_type not in self._mapping:
            return []
        
        changed_files = []
        kernels = self._mapping[op_type]._priority_list()
        
        for kernel_class in kernels:
            # Create a dummy instance to check shared files
            try:
                # Get a minimal config to instantiate the kernel
                dummy_config = {}
                for field in kernel_class.__dataclass_fields__:
                    if hasattr(kernel_class, field):
                        field_type = kernel_class.__dataclass_fields__[field].type
                        if field_type == str:
                            dummy_config[field] = "dummy"
                        elif field_type == int:
                            dummy_config[field] = 0
                        elif field_type == list:
                            dummy_config[field] = []
                
                kernel_instance = kernel_class(**dummy_config)
                files = cache_manager.check_shared_files_changed(kernel_instance)
                changed_files.extend(files)
            except:
                # If we can't instantiate the kernel, skip it
                pass
        
        return list(set(changed_files))  # Remove duplicates

#####################################################
## The Global Kernel Registry
#####################################################
gkr = KernelRegistry()
