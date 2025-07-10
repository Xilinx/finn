"""
Kernel cache manager for storing and retrieving generated kernel files.

Provides caching functionality to avoid regenerating kernels with identical
configurations, significantly reducing build times for HLS kernels.
"""

import hashlib
import json
import os
import shutil
import time
from multiprocessing import Lock
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util

from .kernel import Kernel


class CacheManager:
    """Manages instance file caching based on configuration hashes."""

    # Thread lock for metadata operations
    _metadata_lock = Lock()
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Custom cache directory path. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.finnkernel_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for cache information
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Session statistics for tracking hits/misses
        self.session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_details": [],  # List of (kernel_name, kernel_class, op_type)
            "miss_details": []  # List of (kernel_name, kernel_class, op_type)
        }
    
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                # Create fresh metadata if loading fails
                self.metadata = {
                    "version": "1.0",
                    "entries": {},
                    "created": time.time()
                }
        else:
            self.metadata = {
                "version": "1.0",
                "entries": {},
                "created": time.time()
            }
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            # Reload metadata before saving to merge any concurrent changes
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                    # Merge entries from existing metadata
                    existing_entries = existing_metadata.get("entries", {})
                    current_entries = self.metadata.get("entries", {})
                    # Combine entries (current takes precedence for conflicts)
                    merged_entries = {**existing_entries, **current_entries}
                    self.metadata["entries"] = merged_entries
                except Exception as e:
                    print(f"Error: Could not merge existing metadata: {e}")
            
            # Use atomic write to prevent corruption
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            # Atomic move
            temp_file.replace(self.metadata_file)
        except Exception as e:
            print(f"Error: Error saving metadata to {self.metadata_file}: {e}")
    
    def _get_shared_files_hashes(self, kernel: Kernel) -> Dict[str, str]:
        """Get hashes of shared files referenced by the kernel.
        
        Args:
            kernel: Kernel instance
            
        Returns:
            Dictionary mapping shared file paths to their content hashes
        """
        file_hashes = {}
        
        # Get the base directory (typically where finnkernel is installed)
        base_dir = Path(__file__).parent
        
        for lib_name, rel_path in kernel.sharedFiles:
            # Resolve the actual path to the shared files
            if lib_name == "finn-hlslib":
                # finn-hlslib is in deps/finn-hlslib
                shared_dir = base_dir / "deps" / "finn-hlslib"
            else:
                # For other libraries, assume they're relative to finnkernel
                shared_dir = base_dir / lib_name
            
            actual_path = shared_dir / rel_path
            
            if actual_path.exists():
                if actual_path.is_file():
                    # Single file - hash its content
                    file_hashes[str(actual_path)] = self._hash_file(actual_path)
                elif actual_path.is_dir():
                    # Directory - hash all relevant files recursively
                    dir_hashes = self._hash_directory(actual_path)
                    file_hashes.update(dir_hashes)
            else:
                # File doesn't exist - record as missing (will invalidate cache if file appears later)
                file_hashes[str(actual_path)] = "MISSING"
        
        return file_hashes
    
    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file's content.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash as hex string
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "ERROR"
    
    def _hash_directory(self, dir_path: Path, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        """Recursively hash all files in a directory.
        
        Args:
            dir_path: Directory to hash
            extensions: List of file extensions to include (e.g., ['.hpp', '.cpp'])
                       If None, includes common HLS header extensions
            
        Returns:
            Dictionary mapping file paths to their hashes
        """
        if extensions is None:
            extensions = ['.hpp', '.cpp', '.h', '.c', '.tcl']
        
        file_hashes = {}
        
        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    file_hashes[str(file_path)] = self._hash_file(file_path)
        except Exception:
            pass
        
        return file_hashes
    
    def _compute_hash(self, kernel: Kernel, config: Dict[str, Any]) -> str:
        """Compute hash for kernel and configuration.
        
        Args:
            kernel: Kernel instance
            config: Configuration dictionary
            
        Returns:
            SHA256 hash string
        """
        # Get kernel class-level properties that affect compilation
        kernel_class_data = self._get_kernel_class_data(kernel)
        
        # Create hashable representation
        hash_data = {
            "kernel_class": kernel.__class__.__name__,
            "kernel_module": kernel.__class__.__module__,
            "config": config,
            "shared_files": self._get_shared_files_hashes(kernel),
            "kernel_class_data": kernel_class_data
        }
        
        # Convert to JSON string with sorted keys for consistent hashing
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        
        # Compute SHA256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_kernel_class_data(self, kernel: Kernel) -> Dict[str, Any]:
        """Get class-level data that affects kernel compilation.
        
        Args:
            kernel: Kernel instance
            
        Returns:
            Dictionary of class-level properties
        """
        class_data = {}
        
        # Get constraints if they exist
        if hasattr(kernel.__class__, '_constraints'):
            # Convert lambda functions to a deterministic representation
            constraints = kernel.__class__._constraints
            if constraints:
                # Use the module and qualname to create a deterministic representation
                constraint_repr = []
                for constraint in constraints:
                    if hasattr(constraint, '__module__') and hasattr(constraint, '__qualname__'):
                        constraint_repr.append(f"{constraint.__module__}.{constraint.__qualname__}")
                    else:
                        # Fallback for functions without proper qualname
                        constraint_repr.append(f"{kernel.__class__.__module__}.{kernel.__class__.__name__}.<lambda>")
                class_data['_constraints'] = constraint_repr
        
        # Get shared files definition
        if hasattr(kernel.__class__, 'sharedFiles'):
            shared_files = kernel.__class__.sharedFiles
            # Convert to serializable format
            class_data['sharedFiles'] = [(lib, str(path)) for lib, path in shared_files]
        
        # Get kernel files definition
        if hasattr(kernel.__class__, 'kernelFiles'):
            kernel_files = kernel.__class__.kernelFiles
            class_data['kernelFiles'] = [str(path) for path in kernel_files]
        
        # Get implementation style
        if hasattr(kernel.__class__, 'impl_style'):
            class_data['impl_style'] = kernel.__class__.impl_style
        
        return class_data
    
    def _record_cache_hit(self, kernel_name: str, kernel_class: str, op_type: str):
        """Record a cache hit for session statistics."""
        self.session_stats["cache_hits"] += 1
        self.session_stats["hit_details"].append((kernel_name, kernel_class, op_type))
    
    def _record_cache_miss(self, kernel_name: str, kernel_class: str, op_type: str):
        """Record a cache miss for session statistics."""
        self.session_stats["cache_misses"] += 1
        self.session_stats["miss_details"].append((kernel_name, kernel_class, op_type))
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_details": [],
            "miss_details": []
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session cache statistics.
        
        Returns:
            Dictionary with session cache hit/miss statistics
        """
        total_requests = self.session_stats["cache_hits"] + self.session_stats["cache_misses"]
        hit_rate = (self.session_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.session_stats["cache_hits"],
            "cache_misses": self.session_stats["cache_misses"],
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "hit_details": self.session_stats["hit_details"],
            "miss_details": self.session_stats["miss_details"]
        }
    
    def _get_cache_path(self, op_type: str, cache_hash: str) -> Path:
        """Get cache directory path for a given hash.
        
        Args:
            op_type: Operation type (e.g., 'FMPadding')
            cache_hash: Cache hash string
            
        Returns:
            Path to cache directory
        """
        return self.cache_dir / op_type / cache_hash
    
    def has_cached_files(self, kernel: Kernel, config: Dict[str, Any]) -> bool:
        """Check if cached files exist for kernel and configuration.
        
        Args:
            kernel: Kernel instance
            config: Configuration dictionary
            
        Returns:
            True if cached files exist, False otherwise
        """
        cache_hash = self._compute_hash(kernel, config)
        op_type = getattr(kernel, 'op_type', kernel.__class__.__name__)
        cache_path = self._get_cache_path(op_type, cache_hash)
        
        kernel_name = getattr(kernel, 'name', f"{kernel.__class__.__name__}_unnamed")
        kernel_class = kernel.__class__.__name__
        
        # Check if cache directory exists and has files
        if not cache_path.exists():
            self._record_cache_miss(kernel_name, kernel_class, op_type)
            return False
        
        with CacheManager._metadata_lock:
            self._load_metadata()
        # Check if cache entry is in metadata
        if cache_hash not in self.metadata["entries"]:
            self._record_cache_miss(kernel_name, kernel_class, op_type)
            return False
        
        # Verify cache entry hasn't expired (basic staleness check)
        entry = self.metadata["entries"][cache_hash]
        if entry.get("expired", False):
            self._record_cache_miss(kernel_name, kernel_class, op_type)
            return False
        
        self._record_cache_hit(kernel_name, kernel_class, op_type)
        return True
    
    def get_cached_files(self, kernel: Kernel, config: Dict[str, Any], 
                        target_dir: Path) -> bool:
        """Retrieve cached files to target directory.
        
        Args:
            kernel: Kernel instance
            config: Configuration dictionary
            target_dir: Directory to copy cached files to
            
        Returns:
            True if files were successfully retrieved, False otherwise
        """
        kernel_name = getattr(kernel, 'name', f"{kernel.__class__.__name__}_unnamed")
        
        if not self.has_cached_files(kernel, config):
            print(f"Error: Cache files not available for {kernel_name}")
            return False
        
        cache_hash = self._compute_hash(kernel, config)
        op_type = getattr(kernel, 'op_type', kernel.__class__.__name__)
        cache_path = self._get_cache_path(op_type, cache_hash)
        
        
        try:
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Check what's actually in the cache directory
            cache_items = list(cache_path.iterdir())
            
            # Copy all files from cache to target
            for item in cache_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir)
                elif item.is_dir():
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
            
            with CacheManager._metadata_lock:
                self._load_metadata()
                # Update access time in metadata
                self.metadata["entries"][cache_hash]["last_accessed"] = time.time()
                self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error: Error retrieving cached files for {kernel_name}: {e}")
            return False
    
    def store_generated_files(self, kernel: Kernel, config: Dict[str, Any], 
                            source_dir: Path) -> bool:
        """Store generated files in cache.
        
        Args:
            kernel: Kernel instance
            config: Configuration dictionary
            source_dir: Directory containing generated files
            
        Returns:
            True if files were successfully stored, False otherwise
        """
        cache_hash = self._compute_hash(kernel, config)
        op_type = getattr(kernel, 'op_type', kernel.__class__.__name__)
        cache_path = self._get_cache_path(op_type, cache_hash)
        kernel_name = getattr(kernel, 'name', f"{kernel.__class__.__name__}_unnamed")
        
        try:
            # Create cache directory
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Check what we're about to store
            source_items = list(source_dir.iterdir())
            
            # Copy all files from source to cache
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, cache_path)
                elif item.is_dir():
                    shutil.copytree(item, cache_path / item.name, dirs_exist_ok=True)
            
            with CacheManager._metadata_lock:
                self._load_metadata()
                # Update metadata
                self.metadata["entries"][cache_hash] = {
                    "op_type": op_type,
                    "kernel_class": kernel.__class__.__name__,
                    "kernel_name": kernel_name,  # Add kernel name for debugging
                    "created": time.time(),
                    "last_accessed": time.time(),
                    "config_hash": cache_hash,
                    "shared_files": self._get_shared_files_hashes(kernel)
                }
                self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error: storing generated files for {kernel_name}: {e}")
            return False
    
    def invalidate_cache(self, op_type: Optional[str] = None, 
                        kernel_class: Optional[str] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            op_type: Operation type to invalidate (if None, invalidates all)
            kernel_class: Kernel class to invalidate (if None, invalidates all)
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        with CacheManager._metadata_lock:
            self._load_metadata()
            for cache_hash, entry in self.metadata["entries"].items():
                should_invalidate = False
                
                # Check type/class filters
                if op_type is not None and entry.get("op_type") != op_type:
                    continue
                
                if kernel_class is not None and entry.get("kernel_class") != kernel_class:
                    continue
                
                # If no filters specified, invalidate all
                if op_type is None and kernel_class is None:
                    should_invalidate = True
                else:
                    should_invalidate = True
                
                if should_invalidate:
                    entry["expired"] = True
                    invalidated += 1
            
            if invalidated > 0:
                self._save_metadata()
        
        return invalidated
    
    def check_shared_files_changed(self, kernel: Kernel) -> List[str]:
        """Check if any shared files referenced by kernel have changed.
        
        Args:
            kernel: Kernel instance
            
        Returns:
            List of changed file paths (empty if no changes)
        """
        current_hashes = self._get_shared_files_hashes(kernel)
        changed_files = []
        
        # Find cache entries for this kernel class
        kernel_class = kernel.__class__.__name__
        
        with CacheManager._metadata_lock:
            self._load_metadata()
        for cache_hash, entry in self.metadata["entries"].items():
            if (entry.get("kernel_class") == kernel_class and 
                not entry.get("expired", False)):
                
                stored_hashes = entry.get("shared_files", {})
                
                # Check for changed files
                for file_path, current_hash in current_hashes.items():
                    stored_hash = stored_hashes.get(file_path)
                    if stored_hash != current_hash:
                        changed_files.append(file_path)
                
                # Check for new files
                for file_path in stored_hashes:
                    if file_path not in current_hashes:
                        changed_files.append(file_path)
        
        return list(set(changed_files))  # Remove duplicates
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries from disk.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        
        with CacheManager._metadata_lock:
            self._load_metadata()
            for cache_hash, entry in list(self.metadata["entries"].items()):
                if entry.get("expired", False):
                    op_type = entry.get("op_type", "unknown")
                    cache_path = self._get_cache_path(op_type, cache_hash)
                    
                    try:
                        if cache_path.exists():
                            shutil.rmtree(cache_path)
                        del self.metadata["entries"][cache_hash]
                        cleaned += 1
                    except Exception as e:
                        print(f"Error cleaning up cache entry {cache_hash}: {e}")
            
            if cleaned > 0:
                self._save_metadata()
        
        return cleaned
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with CacheManager._metadata_lock:
            self._load_metadata()
        total_entries = len(self.metadata["entries"])
        expired_entries = sum(1 for e in self.metadata["entries"].values() 
                            if e.get("expired", False))
        active_entries = total_entries - expired_entries
        
        # Calculate cache size
        cache_size = 0
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    cache_size += os.path.getsize(os.path.join(root, file))
        except:
            cache_size = -1
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_size_bytes": cache_size,
            "cache_size_mb": cache_size / (1024 * 1024) if cache_size >= 0 else -1
        }


# Global cache manager instance
cache_manager = CacheManager()
