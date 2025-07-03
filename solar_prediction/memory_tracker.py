"""
Memory tracking utility for PyTorch models.

This module provides utilities to track GPU memory usage during training and evaluation,
particularly useful for memory-efficient model operations.
"""

import torch
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    allocated: float  # MB
    cached: float    # MB
    max_allocated: float  # MB
    timestamp: str
    description: str = ""


class MemoryTracker:
    """
    Utility class to track GPU memory usage during model operations.
    
    Features:
    - Track memory before/after operations
    - Log memory usage with configurable verbosity
    - Context manager for automatic tracking
    - Memory cleanup utilities
    """
    
    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        """
        Initialize memory tracker.
        
        Args:
            device: Device to track ('cuda', 'cpu', or specific device)
            verbose: Whether to log detailed memory information
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.logger = logging.getLogger(f'MemoryTracker.{self.device}')
        
        # Configure logger
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
    def is_cuda_available(self) -> bool:
        """Check if CUDA device is available for tracking."""
        return torch.cuda.is_available() and 'cuda' in str(self.device)
        
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory information in MB.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.is_cuda_available():
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'max_allocated': 0.0
            }
        
        # Convert bytes to MB
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        return {
            'allocated': allocated,
            'cached': cached,
            'max_allocated': max_allocated
        }
    
    def snapshot(self, name: str, description: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot with a given name.
        
        Args:
            name: Identifier for this snapshot
            description: Optional description of what's happening
            
        Returns:
            MemorySnapshot object
        """
        from datetime import datetime
        
        memory_info = self.get_memory_info()
        snapshot = MemorySnapshot(
            allocated=memory_info['allocated'],
            cached=memory_info['cached'],
            max_allocated=memory_info['max_allocated'],
            timestamp=datetime.now().strftime('%H:%M:%S'),
            description=description
        )
        
        self.snapshots[name] = snapshot
        
        if self.verbose:
            self.logger.debug(
                f"Memory snapshot '{name}': {snapshot.allocated:.1f}MB allocated, "
                f"{snapshot.cached:.1f}MB cached {description}"
            )
        
        return snapshot
    
    def log_memory_diff(self, before_name: str, after_name: str, 
                       operation_name: str = "operation") -> None:
        """
        Log the difference in memory usage between two snapshots.
        
        Args:
            before_name: Name of the 'before' snapshot
            after_name: Name of the 'after' snapshot  
            operation_name: Name of the operation for logging
        """
        if before_name not in self.snapshots or after_name not in self.snapshots:
            self.logger.warning(f"Cannot find snapshots for memory diff: {before_name}, {after_name}")
            return
            
        before = self.snapshots[before_name]
        after = self.snapshots[after_name]
        
        allocated_diff = after.allocated - before.allocated
        cached_diff = after.cached - before.cached
        
        log_level = logging.INFO
        if allocated_diff > 100:  # More than 100MB increase
            log_level = logging.WARNING
            
        self.logger.log(
            log_level,
            f"{operation_name} memory change: "
            f"allocated {allocated_diff:+.1f}MB ({before.allocated:.1f} → {after.allocated:.1f}), "
            f"cached {cached_diff:+.1f}MB ({before.cached:.1f} → {after.cached:.1f})"
        )
    
    def cleanup_memory(self, force: bool = False) -> None:
        """
        Clean up GPU memory.
        
        Args:
            force: Whether to force garbage collection
        """
        if not self.is_cuda_available():
            return
            
        # Empty cache
        torch.cuda.empty_cache()
        
        if force:
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # Call again after gc
            
        if self.verbose:
            memory_info = self.get_memory_info()
            self.logger.debug(
                f"Memory cleanup completed. Current usage: "
                f"{memory_info['allocated']:.1f}MB allocated, "
                f"{memory_info['cached']:.1f}MB cached"
            )
    
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        if self.is_cuda_available():
            torch.cuda.reset_peak_memory_stats()
            if self.verbose:
                self.logger.debug("Peak memory stats reset")
    
    @contextmanager
    def track_memory(self, operation_name: str, cleanup_after: bool = True):
        """
        Context manager to automatically track memory for an operation.
        
        Args:
            operation_name: Name of the operation being tracked
            cleanup_after: Whether to cleanup memory after the operation
            
        Usage:
            with memory_tracker.track_memory("training_epoch"):
                # Your code here
                pass
        """
        before_name = f"{operation_name}_before"
        after_name = f"{operation_name}_after"
        
        # Take before snapshot
        self.snapshot(before_name, f"before {operation_name}")
        
        try:
            yield self
        finally:
            # Take after snapshot and log difference
            self.snapshot(after_name, f"after {operation_name}")
            self.log_memory_diff(before_name, after_name, operation_name)
            
            # Optional cleanup
            if cleanup_after:
                self.cleanup_memory()
    
    def get_summary(self) -> str:
        """
        Get a summary of all memory snapshots.
        
        Returns:
            Formatted string with memory tracking summary
        """
        if not self.snapshots:
            return "No memory snapshots recorded."
        
        summary_lines = ["Memory Tracking Summary:"]
        summary_lines.append("-" * 50)
        
        for name, snapshot in self.snapshots.items():
            summary_lines.append(
                f"{snapshot.timestamp} | {name:20} | "
                f"Allocated: {snapshot.allocated:6.1f}MB | "
                f"Cached: {snapshot.cached:6.1f}MB | "
                f"{snapshot.description}"
            )
        
        return "\n".join(summary_lines)
    
    def log_device_info(self) -> None:
        """Log information about the current device."""
        if self.is_cuda_available():
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            self.logger.info(
                f"GPU Device Info: {device_name} (Device {current_device}/{device_count-1}), "
                f"Total Memory: {total_memory:.1f}GB"
            )
        else:
            self.logger.info("Using CPU - no GPU memory tracking available")
