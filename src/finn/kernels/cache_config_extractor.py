"""
Utility functions for extracting kernel configurations for caching.

This module provides standardized config extraction logic used by both
the HLS builder and test code to ensure consistent hashing.
"""

import dataclasses
from typing import Any, Dict, Set

def extract_kernel_config_for_cache(kernel) -> Dict[str, Any]:
    """Extract kernel configuration for cache hashing.
    
    This function automatically extracts all user-facing dataclass fields
    from any kernel type, filtering out internal/runtime fields that don't
    affect the generated output.
    
    Args:
        kernel: Kernel instance (any dataclass-based kernel)
        
    Returns:
        Dictionary containing kernel configuration for cache hashing
    """
    kernel_config = {}
    
    # Fields to exclude from cache key (internal/runtime fields that don't affect output)
    excluded_fields = _get_excluded_fields()
    
    # Extract all dataclass fields except excluded ones
    if hasattr(kernel, '__dataclass_fields__'):
        for field_name, field_info in kernel.__dataclass_fields__.items():
            # Skip excluded fields
            if field_name in excluded_fields:
                continue
                
            # Skip private fields (starting with underscore)
            if field_name.startswith('_'):
                continue
                
            # Extract the field value if it exists
            if hasattr(kernel, field_name):
                value = getattr(kernel, field_name)
                
                # Convert complex types to hashable representations
                kernel_config[field_name] = _make_hashable(value)
    
    # Always include critical fields that might be properties/computed
    critical_fields = ['name', 'len_node_input', 'len_node_output']
    for field_name in critical_fields:
        if field_name not in kernel_config and hasattr(kernel, field_name):
            value = getattr(kernel, field_name)
            kernel_config[field_name] = _make_hashable(value)
    
    # Add kernel class information for disambiguation
    kernel_config['__kernel_class__'] = kernel.__class__.__name__
    kernel_config['__kernel_module__'] = kernel.__class__.__module__
    
    return kernel_config


def _get_excluded_fields() -> Set[str]:
    """Get set of field names to exclude from cache key generation.
    
    These are typically internal, runtime, or non-deterministic fields
    that don't affect the generated kernel output.
    
    Returns:
        Set of field names to exclude
    """
    return {
        # Internal/runtime fields
        '_internal_state',
        '_cached_data', 
        '_temp_files',
        '_build_context',
        '_logger',
        
        # Timing/performance fields that don't affect output
        'build_time',
        'last_accessed',
        'access_count',
        
        # Temporary/session-specific fields
        'session_id',
        'build_id',
        'temp_dir',
        'work_dir',
        
        # Add more as needed - these can be extended per kernel type
    }


def _make_hashable(value: Any) -> Any:
    """Convert a value to a hashable representation for cache keys.
    
    Handles common non-hashable types that might appear in kernel configs.
    
    Args:
        value: The value to make hashable
        
    Returns:
        Hashable representation of the value
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    elif isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    elif isinstance(value, set):
        return tuple(sorted(_make_hashable(item) for item in value))
    elif hasattr(value, '__dict__'):
        # Handle objects by converting to dict representation
        return _make_hashable(value.__dict__)
    elif dataclasses.is_dataclass(value):
        # Handle nested dataclasses
        return tuple(sorted((f.name, _make_hashable(getattr(value, f.name))) 
                          for f in dataclasses.fields(value) 
                          if not f.name.startswith('_')))
    else:
        # Fallback: convert to string representation
        return str(value)