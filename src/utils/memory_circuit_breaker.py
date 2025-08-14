"""
Simple memory circuit breaker for Slurm environments.

Provides accurate memory monitoring against Slurm allocations and implements
a simple circuit breaker pattern to prevent memory-related job crashes.
"""

import os
import time
import logging
import psutil


def get_slurm_memory_allocation_gb():
    """Get the job's memory allocation from Slurm environment variables."""
    # Try different Slurm memory variables (in order of preference)
    mem_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mem_mb:
        try:
            return int(mem_mb) / 1024  # Convert MB to GB
        except ValueError:
            pass
    
    # Alternative: memory per CPU * number of CPUs
    mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
    cpus = os.environ.get('SLURM_CPUS_ON_NODE')
    if mem_per_cpu and cpus:
        try:
            return (int(mem_per_cpu) * int(cpus)) / 1024  # MB to GB
        except ValueError:
            pass
    
    # Fallback: use system memory (not ideal but better than crashing)
    system_mem_gb = psutil.virtual_memory().total / (1024**3)
    logging.warning(f"No Slurm memory allocation found, using system memory: {system_mem_gb:.1f}GB")
    return system_mem_gb


def get_current_memory_usage_gb():
    """Get current RSS memory usage for main process + all child workers."""
    try:
        main_process = psutil.Process(os.getpid())
        
        # Get RSS (Resident Set Size) - actual physical memory used
        total_rss = main_process.memory_info().rss
        
        # Add all child processes (your workers)
        for child in main_process.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue  # Process may have terminated
        
        return total_rss / (1024**3)  # Convert bytes to GB
        
    except Exception as e:
        logging.warning(f"Error getting memory usage: {e}")
        return 0.0


def get_total_process_memory_usage_gb():
    """Get total memory usage across all processes in the job.
    
    This function works from any process (main or worker) by finding
    all processes that belong to the same job/session.
    """
    try:
        current_process = psutil.Process(os.getpid())
        
        # Get the session ID to find related processes
        try:
            session_id = os.getsid(current_process.pid)
        except (OSError, AttributeError):
            # Fallback: just return current process memory
            return current_process.memory_info().rss / (1024**3)
        
        total_rss = 0
        
        # Sum memory usage of all processes in the same session
        for proc in psutil.process_iter(['pid', 'memory_info']):
            try:
                if os.getsid(proc.info['pid']) == session_id:
                    total_rss += proc.info['memory_info'].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue  # Process may have terminated or no access
        
        return total_rss / (1024**3)  # Convert bytes to GB
        
    except Exception as e:
        logging.warning(f"Error getting total process memory usage: {e}")
        # Fallback to just current process
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        except:
            return 0.0


def get_worker_process_memory_usage_gb():
    """Get memory usage for just the current worker process.
    
    This function is specifically for worker processes to check their own
    memory usage without being affected by other processes in the session.
    Should be used by worker processes for self-monitoring.
    """
    try:
        current_process = psutil.Process(os.getpid())
        return current_process.memory_info().rss / (1024**3)
    except Exception as e:
        logging.warning(f"Error getting worker process memory usage: {e}")
        return 0.0


def memory_circuit_breaker(high_threshold_pct=85, low_threshold_pct=75):
    """
    Pause execution when memory usage exceeds Slurm allocation limits.
    
    Args:
        high_threshold_pct: Pause when memory exceeds this % of allocation
        low_threshold_pct: Resume when memory drops below this % of allocation
    """
    allocation_gb = get_slurm_memory_allocation_gb()
    
    current_gb = get_current_memory_usage_gb()
    current_pct = (current_gb / allocation_gb) * 100
    
    # Check if we need to pause
    if current_pct <= high_threshold_pct:
        return  # Safe to proceed
    
    # Memory is too high - enter circuit breaker mode
    logging.warning(f"MEMORY CIRCUIT BREAKER ACTIVATED!")
    logging.warning(f"Current usage: {current_pct:.1f}% ({current_gb:.1f}GB / {allocation_gb:.1f}GB allocated)")
    logging.warning(f"Pausing new work until memory drops below {low_threshold_pct}%...")
    
    # Wait for memory to drop
    while True:
        time.sleep(5)  # Wait 5 seconds for writes/cleanup
        
        current_gb = get_current_memory_usage_gb()
        current_pct = (current_gb / allocation_gb) * 100
        
        if current_pct <= low_threshold_pct:
            logging.info(f"Memory dropped to {current_pct:.1f}% - resuming processing")
            return
        
        # Still too high, keep waiting
        logging.info(f"Memory still high: {current_pct:.1f}% - continuing to wait...")


def debug_slurm_memory():
    """Debug Slurm memory environment variables."""
    print("Slurm Memory Environment Variables:")
    print("=" * 50)
    
    slurm_vars = [var for var in os.environ.keys() if 'SLURM' in var and 'MEM' in var]
    if slurm_vars:
        for var in slurm_vars:
            print(f"{var} = {os.environ[var]}")
    else:
        print("No Slurm memory variables found")
    
    print(f"\nCurrent allocation: {get_slurm_memory_allocation_gb():.1f}GB")
    print(f"Current usage: {get_current_memory_usage_gb():.1f}GB")
    current_pct = (get_current_memory_usage_gb() / get_slurm_memory_allocation_gb()) * 100
    print(f"Usage percentage: {current_pct:.1f}%")


def log_memory_status(context=""):
    """Log current memory status for debugging."""
    allocation_gb = get_slurm_memory_allocation_gb()
    current_gb = get_current_memory_usage_gb()
    current_pct = (current_gb / allocation_gb) * 100
    
    context_str = f" ({context})" if context else ""
    logging.info(f"Memory status{context_str}: {current_pct:.1f}% ({current_gb:.1f}GB / {allocation_gb:.1f}GB)")
