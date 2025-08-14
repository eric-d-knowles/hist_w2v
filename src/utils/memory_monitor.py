#!/usr/bin/env python3
"""
Comprehensive memory monitoring utility for tracking system and process memory.

This module provides tools to monitor memory usage across the system and
specific processes, especially useful during intensive operations like
ngram processing with multiple workers.
"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import psutil


def get_system_memory_info() -> Dict[str, float]:
    """
    Get comprehensive system memory information.
    
    Returns:
        Dict containing memory info in GB:
        - total: Total system memory
        - available: Available memory
        - used: Used memory
        - percent: Memory usage percentage
        - free: Free memory
    """
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024**3),
        'available': memory.available / (1024**3),
        'used': memory.used / (1024**3),
        'percent': memory.percent,
        'free': memory.free / (1024**3)
    }


def get_process_memory_info(pid: int) -> Optional[Dict[str, float]]:
    """
    Get memory information for a specific process.
    
    Args:
        pid: Process ID
        
    Returns:
        Dict containing process memory info in GB, or None if process not found
    """
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / (1024**3),
            'vms': memory_info.vms / (1024**3),
            'percent': process.memory_percent()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def get_all_python_processes() -> List[Dict[str, any]]:
    """
    Get information about all Python processes in the system.
    
    Returns:
        List of dicts containing process info
    """
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                memory_gb = proc.info['memory_info'].rss / (1024**3)
                python_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': ' '.join(proc.info['cmdline'][:3]),  # First 3 args
                    'memory_gb': memory_gb
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return sorted(python_processes, key=lambda x: x['memory_gb'], reverse=True)


def get_child_processes_memory(parent_pid: int) -> Tuple[List[int], float]:
    """
    Get total memory usage of all child processes.
    
    Args:
        parent_pid: Parent process ID
        
    Returns:
        Tuple of (list of child PIDs, total memory in GB)
    """
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        
        child_pids = []
        total_memory = 0.0
        
        for child in children:
            try:
                child_pids.append(child.pid)
                total_memory += child.memory_info().rss / (1024**3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return child_pids, total_memory
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [], 0.0


def monitor_memory_usage(
    duration: int = 60,
    interval: int = 5,
    output_file: Optional[str] = None
) -> None:
    """
    Monitor system memory usage over time.
    
    Args:
        duration: Total monitoring duration in seconds
        interval: Sampling interval in seconds
        output_file: Optional file to write results to
    """
    print(f"Monitoring memory for {duration} seconds at {interval}s intervals...")
    
    start_time = time.time()
    current_pid = os.getpid()
    
    results = []
    
    while time.time() - start_time < duration:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # System memory
        sys_mem = get_system_memory_info()
        
        # Current process memory
        proc_mem = get_process_memory_info(current_pid)
        
        # Child processes memory
        child_pids, child_memory = get_child_processes_memory(current_pid)
        
        result = {
            'timestamp': timestamp,
            'system_used_gb': sys_mem['used'],
            'system_percent': sys_mem['percent'],
            'process_memory_gb': proc_mem['rss'] if proc_mem else 0,
            'child_count': len(child_pids),
            'child_memory_gb': child_memory,
            'total_process_memory_gb': (proc_mem['rss'] if proc_mem else 0) + child_memory
        }
        
        results.append(result)
        
        print(f"{timestamp} | System: {sys_mem['used']:.1f}GB ({sys_mem['percent']:.1f}%) | "
              f"Process: {proc_mem['rss'] if proc_mem else 0:.2f}GB | "
              f"Children: {len(child_pids)} processes, {child_memory:.2f}GB | "
              f"Total: {result['total_process_memory_gb']:.2f}GB")
        
        time.sleep(interval)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write("timestamp,system_used_gb,system_percent,process_memory_gb,"
                   "child_count,child_memory_gb,total_process_memory_gb\n")
            for result in results:
                f.write(f"{result['timestamp']},{result['system_used_gb']:.3f},"
                       f"{result['system_percent']:.1f},{result['process_memory_gb']:.3f},"
                       f"{result['child_count']},{result['child_memory_gb']:.3f},"
                       f"{result['total_process_memory_gb']:.3f}\n")
        print(f"Results saved to {output_file}")


def check_memory_pressure(threshold_percent: float = 85.0) -> Dict[str, any]:
    """
    Check if system is under memory pressure.
    
    Args:
        threshold_percent: Memory usage threshold to consider as pressure
        
    Returns:
        Dict with pressure status and recommendations
    """
    sys_mem = get_system_memory_info()
    python_procs = get_all_python_processes()
    
    is_under_pressure = sys_mem['percent'] > threshold_percent
    
    result = {
        'under_pressure': is_under_pressure,
        'current_usage_percent': sys_mem['percent'],
        'threshold_percent': threshold_percent,
        'available_gb': sys_mem['available'],
        'top_python_processes': python_procs[:10],  # Top 10 by memory
        'recommendations': []
    }
    
    if is_under_pressure:
        result['recommendations'].extend([
            "Consider reducing the number of worker processes",
            "Implement more frequent garbage collection",
            "Use smaller batch sizes for processing",
            "Monitor for memory leaks in worker processes"
        ])
    
    if sys_mem['available'] < 10.0:  # Less than 10GB available
        result['recommendations'].append(
            "CRITICAL: Very low available memory, consider stopping non-essential processes"
        )
    
    return result


def force_garbage_collection() -> Dict[str, int]:
    """
    Force garbage collection and return statistics.
    
    Returns:
        Dict with collection statistics
    """
    before_counts = [len(gc.get_objects())]
    
    # Force collection of all generations
    collected = []
    for generation in range(3):
        collected.append(gc.collect(generation))
    
    after_counts = [len(gc.get_objects())]
    
    return {
        'objects_before': before_counts[0],
        'objects_after': after_counts[0],
        'objects_freed': before_counts[0] - after_counts[0],
        'generation_0_collected': collected[0],
        'generation_1_collected': collected[1],
        'generation_2_collected': collected[2],
        'total_collected': sum(collected)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory monitoring utility")
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in seconds (default: 60)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Sampling interval in seconds (default: 5)')
    parser.add_argument('--output', type=str,
                       help='Output file for results (optional)')
    parser.add_argument('--check-pressure', action='store_true',
                       help='Check current memory pressure and exit')
    parser.add_argument('--list-python', action='store_true',
                       help='List all Python processes and exit')
    parser.add_argument('--gc', action='store_true',
                       help='Force garbage collection and show stats')
    
    args = parser.parse_args()
    
    if args.check_pressure:
        pressure = check_memory_pressure()
        print(f"Memory pressure check:")
        print(f"  Under pressure: {pressure['under_pressure']}")
        print(f"  Current usage: {pressure['current_usage_percent']:.1f}%")
        print(f"  Available: {pressure['available_gb']:.1f}GB")
        if pressure['recommendations']:
            print("  Recommendations:")
            for rec in pressure['recommendations']:
                print(f"    - {rec}")
    
    elif args.list_python:
        procs = get_all_python_processes()
        print("Python processes by memory usage:")
        for proc in procs:
            print(f"  PID {proc['pid']}: {proc['memory_gb']:.3f}GB - {proc['cmdline']}")
    
    elif args.gc:
        stats = force_garbage_collection()
        print("Garbage collection results:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        monitor_memory_usage(args.duration, args.interval, args.output)
