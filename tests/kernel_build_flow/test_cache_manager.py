"""
Tests for the kernel cache manager functionality.

These tests verify that the caching system works correctly for storing
and retrieving generated kernel files based on configuration hashes.
Uses real HLS builder functions to generate actual kernel outputs.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path

from finn.kernels import gkr

from finn.transformation.fpgadataflow.hls_code_builder import gen_hls_node
from finn.util.context import Context


class TestCacheManager:
    """Test suite for CacheManager functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Import global cache manager
        from finn.kernels.cache_manager import cache_manager as global_cache
        self.cache_manager = global_cache
    
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compute_hash_consistency_fmpadding_rect(self):
        """Test that hash computation is consistent for identical FMPadding rect configs."""
        config = {
            "ImgDim": [28, 28],
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 64,
            "SIMD": 8,
            "inputDataType": "INT8",
            "numInputVectors": 4,
            "name": "FMPadding_0",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel1 = gkr.kernel("FMPadding", config)
        kernel2 = gkr.kernel("FMPadding", config)
        
        # Skip test if no suitable kernel found
        if kernel1 is None or kernel2 is None:
            pytest.skip("No suitable FMPadding kernel found in registry")
        
        hash1 = self.cache_manager._compute_hash(kernel1, config)
        hash2 = self.cache_manager._compute_hash(kernel2, config)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_compute_hash_different_configs(self):
        """Test that different configurations produce different hashes."""
        config1 = {
            "ImgDim": [28, 28],
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 64,
            "SIMD": 8,
            "inputDataType": "INT8",
            "numInputVectors": 4,
            "name": "FMPadding_0",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        config2 = {
            "ImgDim": [32, 32],  
            "Padding": [2, 2, 2, 2], 
            "NumChannels": 128,  
            "SIMD": 16,  
            "inputDataType": "INT16", 
            "numInputVectors": 8, 
            "name": "FMPadding_1",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel1 = gkr.kernel("FMPadding", config1)
        kernel2 = gkr.kernel("FMPadding", config2)
        
        # Skip test if no suitable kernels found
        if kernel1 is None or kernel2 is None:
            pytest.skip("No suitable FMPadding kernels found in registry")
        
        hash1 = self.cache_manager._compute_hash(kernel1, config1)
        hash2 = self.cache_manager._compute_hash(kernel2, config2)
        
        assert hash1 != hash2
    
    def test_compute_hash_different_kernel_types(self):
        """Test that different kernel types produce different hashes."""
        # Config for square kernel (square ImgDim and equal padding)
        config_square = {
            "ImgDim": [28, 28],  # Square image
            "Padding": [1, 1, 1, 1],  
            "NumChannels": 64,
            "SIMD": 8,
            "inputDataType": "INT8",
            "numInputVectors": 4,
            "name": "FMPadding_Square",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        # Config for rect kernel (non-square ImgDim)
        config_rect = {
            "ImgDim": [28, 32],  # Non-square image
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 64,
            "SIMD": 8,
            "inputDataType": "INT8",
            "numInputVectors": 4,
            "name": "FMPadding_Rect",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel_square = gkr.kernel("FMPadding", config_square)
        kernel_rect = gkr.kernel("FMPadding", config_rect)
        
        # Skip test if no suitable kernels found
        if kernel_square is None or kernel_rect is None:
            pytest.skip("No suitable FMPadding kernels found in registry")
        
        hash_square = self.cache_manager._compute_hash(kernel_square, config_square)
        hash_rect = self.cache_manager._compute_hash(kernel_rect, config_rect)
        
        assert hash_square != hash_rect
    
    def test_has_cached_files_empty_cache(self):
        """Test checking for cached files when cache is empty."""
        config = {
            "ImgDim": [28, 32],  # Non-square to get rect kernel
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 64,
            "SIMD": 8,
            "inputDataType": "INT8",
            "numInputVectors": 4,
            "name": "FMPadding_0",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel = gkr.kernel("FMPadding", config)
        if kernel is None:
            pytest.skip("No suitable FMPadding kernel found in registry")
        
        assert not self.cache_manager.has_cached_files(kernel, config)
    
    def test_cache_stats(self):
        """Test cache statistics functionality."""
        stats = self.cache_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "cache_dir" in stats
        assert "total_entries" in stats
        assert "active_entries" in stats
        assert "expired_entries" in stats
        assert "cache_size_bytes" in stats
        assert "cache_size_mb" in stats
        
        # Just verify the stats are non-negative integers (global cache may have entries)
        assert stats["total_entries"] >= 0
        assert stats["active_entries"] >= 0
        assert stats["expired_entries"] >= 0
        assert stats["total_entries"] == stats["active_entries"] + stats["expired_entries"]


@pytest.mark.skipif(shutil.which("vitis_hls") is None, reason="vitis_hls not available")
class TestCacheManagerWithHLS:
    """Test suite for CacheManager with real HLS generation."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Import global cache manager
        from finn.kernels.cache_manager import cache_manager as global_cache
        self.cache_manager = global_cache
        
        # Create output directory for HLS generation
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
    
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_store_and_retrieve_real_hls_files(self):
        """Test storing and retrieving cached files generated by real HLS builder.
           Give it a unqiue name each time.
        """
        import uuid
        unique_name = f"FMPadding_Test_{uuid.uuid4().hex[:8]}"
        
        config = {
            "ImgDim": [8, 12],
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 4,
            "SIMD": 2,
            "inputDataType": "INT8",
            "numInputVectors": 1,
            "name": unique_name,
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel = gkr.kernel("FMPadding", config)
        if kernel is None:
            pytest.skip("No suitable FMPadding kernel found in registry")
        
        libraries = {
            "finn-hlslib": Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
        }
        ctx = Context(
            directory=self.output_dir,
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        
        from finn.kernels.cache_manager import cache_manager as global_cache
        from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache
        kernel_config = extract_kernel_config_for_cache(kernel)
        
        initial_cached = global_cache.has_cached_files(kernel, kernel_config)
        
        node_ctx1 = ctx.get_subcontext(Path(kernel.name))
        gen_hls_node(kernel, node_ctx1)
        
        kernel_dir = self.output_dir / kernel.name
        assert kernel_dir.exists()
        assert (kernel_dir / f"{kernel.name}.cpp").exists()
        assert (kernel_dir / f"hls_syn_{kernel.name}.tcl").exists()
        assert (kernel_dir / "ipgen.sh").exists()
        
        project_dir = kernel_dir / f"project_{kernel.name}"
        ip_dir = project_dir / "sol1" / "impl" / "ip"
        assert ip_dir.exists()
        
        assert global_cache.has_cached_files(kernel, kernel_config)
        
        output_dir_2 = Path(self.temp_dir) / "output2"
        output_dir_2.mkdir()
        
        ctx2 = Context(
            directory=output_dir_2,
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        
        node_ctx2 = ctx2.get_subcontext(Path(kernel.name))
        gen_hls_node(kernel, node_ctx2)
        
        kernel_dir_2 = output_dir_2 / kernel.name
        assert kernel_dir_2.exists()
        assert (kernel_dir_2 / f"{kernel.name}.cpp").exists()
        assert (kernel_dir_2 / f"project_{kernel.name}" / "sol1" / "impl" / "ip").exists()
    
    def test_cache_invalidation_on_different_configs(self):
        """Test that different configurations don't use each other's cache."""
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        
        base_config = {
            "ImgDim": [8, 12],  # Non-square to get rect kernel
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 4,
            "SIMD": 2,
            "inputDataType": "INT8",
            "numInputVectors": 1,
            "name": f"FMPadding_Base_{unique_id}",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        # Create modified config
        modified_config = base_config.copy()
        modified_config["NumChannels"] = 8  # Different channel count
        modified_config["name"] = f"FMPadding_Modified_{unique_id}"
        
        kernel_base = gkr.kernel("FMPadding", base_config)
        kernel_modified = gkr.kernel("FMPadding", modified_config)
        
        if kernel_base is None or kernel_modified is None:
            pytest.skip("No suitable FMPadding kernels found in registry")
        
        # Create contexts
        libraries = {
            "finn-hlslib": Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
        }
        
        ctx_base = Context(
            directory=self.output_dir / f"base_{unique_id}",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        ctx_base.directory.mkdir(parents=True, exist_ok=True)
        
        ctx_modified = Context(
            directory=self.output_dir / f"modified_{unique_id}",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        ctx_modified.directory.mkdir(parents=True, exist_ok=True)
        
        # Verify that these configurations are not already cached
        from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache
        base_kernel_config = extract_kernel_config_for_cache(kernel_base)
        modified_kernel_config = extract_kernel_config_for_cache(kernel_modified)
        
        assert not self.cache_manager.has_cached_files(kernel_base, base_kernel_config), \
            "Base configuration should not be cached already"
        assert not self.cache_manager.has_cached_files(kernel_modified, modified_kernel_config), \
            "Modified configuration should not be cached already"
        
        initial_stats = self.cache_manager.get_cache_stats()
        initial_entries = initial_stats["total_entries"]
        
        gen_hls_node(kernel_base, ctx_base)
        
        # Verify first kernel was cached
        assert self.cache_manager.has_cached_files(kernel_base, base_kernel_config), \
            "Base kernel should be cached after generation"
        
        # SHOULD NOT USE THE CACHE 
        gen_hls_node(kernel_modified, ctx_modified)
        
        # Verify second kernel was cached
        assert self.cache_manager.has_cached_files(kernel_modified, modified_kernel_config), \
            "Modified kernel should be cached after generation"
        
        # Both should be cached separately 
        final_stats = self.cache_manager.get_cache_stats()
        assert final_stats["total_entries"] == initial_entries + 2, \
            f"Expected {initial_entries + 2} total entries, got {final_stats['total_entries']}"
        assert final_stats["active_entries"] >= initial_stats["active_entries"] + 2
    
    def test_rect_vs_square_kernel_caching(self):
        """Test that rectangular and square kernels cache separately."""
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        
        # Config for square kernel
        config_square = {
            "ImgDim": [8, 8],  # Square
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 4,
            "SIMD": 2,
            "inputDataType": "INT8",
            "numInputVectors": 1,
            "name": f"FMPadding_Square_{unique_id}",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        # Config for rect kernel  
        config_rect = {
            "ImgDim": [8, 12],  # Non-square
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 4,
            "SIMD": 2,
            "inputDataType": "INT8",
            "numInputVectors": 1,
            "name": f"FMPadding_Rect_{unique_id}",
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel_square = gkr.kernel("FMPadding", config_square)
        kernel_rect = gkr.kernel("FMPadding", config_rect)
        
        if kernel_square is None or kernel_rect is None:
            pytest.skip("No suitable FMPadding kernels found in registry")
        
        # Create contexts
        libraries = {
            "finn-hlslib": Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
        }
        
        ctx_rect = Context(
            directory=self.output_dir / f"rect_{unique_id}",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        ctx_rect.directory.mkdir(parents=True, exist_ok=True)
        
        ctx_square = Context(
            directory=self.output_dir / f"square_{unique_id}",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        ctx_square.directory.mkdir(parents=True, exist_ok=True)
        
        # Extract kernel configurations for checking (same way as HLS builder)
        from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache
        rect_kernel_config = extract_kernel_config_for_cache(kernel_rect)
        square_kernel_config = extract_kernel_config_for_cache(kernel_square)
        
        # Verify that these configurations are not already cached
        assert not self.cache_manager.has_cached_files(kernel_rect, rect_kernel_config), \
            "Rect configuration should not be cached already"
        assert not self.cache_manager.has_cached_files(kernel_square, square_kernel_config), \
            "Square configuration should not be cached already"
        
        initial_stats = self.cache_manager.get_cache_stats()
        initial_entries = initial_stats["total_entries"]
        
        gen_hls_node(kernel_rect, ctx_rect)
        gen_hls_node(kernel_square, ctx_square)
        
        assert self.cache_manager.has_cached_files(kernel_rect, rect_kernel_config), \
            "Rect kernel should be cached after generation"
        assert self.cache_manager.has_cached_files(kernel_square, square_kernel_config), \
            "Square kernel should be cached after generation"
        
        final_stats = self.cache_manager.get_cache_stats()
        assert final_stats["total_entries"] == initial_entries + 2, \
            f"Expected {initial_entries + 2} total entries, got {final_stats['total_entries']}"
        assert final_stats["active_entries"] >= initial_stats["active_entries"] + 2
    
    def test_cache_performance_improvement(self):
        """Test that using cache improves performance (second run should be faster)."""
        import time

        import uuid
        unique_name = f"FMPadding_Perf_{uuid.uuid4().hex[:8]}"
        
        config = {
            "ImgDim": [8, 12],  # Non-square to get rect kernel
            "Padding": [1, 1, 1, 1], 
            "NumChannels": 4,
            "SIMD": 2,
            "inputDataType": "INT8",
            "numInputVectors": 1,
            "name": unique_name,
            "len_node_input": 1,
            "len_node_output": 1
        }
        
        kernel = gkr.kernel("FMPadding", config)
        if kernel is None:
            pytest.skip("No suitable FMPadding kernel found in registry")
        
        # First generation (should be slow - actual HLS synthesis)
        libraries = {
            "finn-hlslib": Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
        }
        
        ctx1 = Context(
            directory=self.output_dir / "perf1",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        node_ctx1 = ctx1.get_subcontext(Path(kernel.name))
        
        start_time = time.time()
        gen_hls_node(kernel, node_ctx1)
        first_duration = time.time() - start_time
        
        # Second generation (USES CACHE)
        ctx2 = Context(
            directory=self.output_dir / "perf2",
            libraries=libraries,
            fpga_part="xc7z020clg400-1",
            clk_ns=10.0,
            clk_hls=10.0
        )
        node_ctx2 = ctx2.get_subcontext(Path(kernel.name))
        
        start_time = time.time()
        gen_hls_node(kernel, node_ctx2)
        second_duration = time.time() - start_time
        
        # Second run should be significantly faster (at least 5x faster)
        print(f"First run: {first_duration:.2f}s, Second run: {second_duration:.2f}s")
        assert second_duration < first_duration / 5, f"Cache didn't improve performance enough: {first_duration:.2f}s -> {second_duration:.2f}s"
        
        # Verify both outputs are identical
        assert (ctx1.directory / kernel.name / f"{kernel.name}.cpp").exists()
        assert (ctx2.directory / kernel.name / f"{kernel.name}.cpp").exists()
        
        content1 = (ctx1.directory / kernel.name / f"{kernel.name}.cpp").read_text()
        content2 = (ctx2.directory / kernel.name / f"{kernel.name}.cpp").read_text()
        assert content1 == content2


class TestKernelRegistryCache:
    """Test cache functionality integrated with kernel registry."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Import global cahce manager
        from finn.kernels.cache_manager import cache_manager as global_cache
        self.cache_manager = global_cache
        
        # Import the global kernel regitry
        from finn.kernels import gkr
        self.gkr = gkr
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_registry_cache_methods(self):
        """Test cache management methods on kernel registry."""
        stats = self.gkr.cache_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        
        cleared = self.gkr.clear_cache()
        assert isinstance(cleared, int)
        
        cleaned = self.gkr.cleanup_cache()
        assert isinstance(cleaned, int)
    
    def test_registry_shared_files_check(self):
        """Test shared files change detection through registry."""
        changed_files = self.gkr.check_shared_files_changed("FMPadding")
        assert isinstance(changed_files, list)
        
        changed_files = self.gkr.check_shared_files_changed("NonExistentOp")
        assert changed_files == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
