"""
System Resource Summary Utility

Provides comprehensive system resource information for debugging and monitoring
Word2GM training environments, including SLURM job allocation details and GPU info.
"""

import os
import socket
import subprocess
import psutil
import pynvml
import re

from IPython.display import display, Markdown
from utils.tf_silence import import_tf_quietly

# Import TensorFlow silently
tf = import_tf_quietly(force_cpu=False)


def get_hostname():
    """Get system hostname."""
    return socket.gethostname()


def get_slurm_info():
    """Get SLURM job allocation information."""
    slurm_info = {}
    
    # CPU allocation
    slurm_info['cpus'] = int(os.environ.get(
        'SLURM_CPUS_PER_TASK', 
        psutil.cpu_count()
    ))
    
    # Memory allocation
    mem_per_node_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mem_per_node_mb:
        slurm_info['memory_gb'] = int(mem_per_node_mb) / 1024
    else:
        slurm_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    
    # Partition information - get both requested and actual
    slurm_info['requested_partitions'] = os.environ.get('SLURM_JOB_PARTITION', 'unknown')
    slurm_info['job_id'] = os.environ.get('SLURM_JOB_ID', 'N/A')
    slurm_info['node_list'] = os.environ.get('SLURM_JOB_NODELIST', 'N/A')
    
    # Get actual partition - try multiple methods
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        # Method 1: Try local squeue command first (might work if we're on a compute node)
        try:
            result = subprocess.run(['squeue', '-j', job_id, '-h', '-o', '%P'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                slurm_info['actual_partition'] = result.stdout.strip()
            else:
                raise subprocess.CalledProcessError(result.returncode, 'squeue')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Method 2: Try SSH with more robust host key handling
            try:
                ssh_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', 
                          '-o', 'UserKnownHostsFile=/dev/null',
                          '-o', 'LogLevel=ERROR',
                          'greene-login', 'squeue', '-j', job_id, '-h', '-o', '%P']
                
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    slurm_info['actual_partition'] = result.stdout.strip()
                else:
                    # Method 3: Try to get info from local SLURM environment variables
                    slurm_info['actual_partition'] = f"SSH failed, using fallback: {slurm_info['requested_partitions']}"
                    
            except subprocess.TimeoutExpired:
                slurm_info['actual_partition'] = f"SSH timeout, using fallback: {slurm_info['requested_partitions']}"
            except Exception as e:
                slurm_info['actual_partition'] = f"SSH error, using fallback: {slurm_info['requested_partitions']}"
    else:
        slurm_info['actual_partition'] = 'N/A (not in SLURM job)'
    
    return slurm_info


def get_gpu_info():
    """Get GPU information using pynvml."""
    gpu_info = {'available': False, 'devices': []}
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info['available'] = device_count > 0
        gpu_info['count'] = device_count
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature if available
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp = None
            
            # Get utilization if available
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
            except:
                gpu_util = None
                memory_util = None
            
            device_info = {
                'index': i,
                'name': name,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_free_gb': memory_info.free / (1024**3),
                'temperature_c': temp,
                'gpu_utilization_percent': gpu_util,
                'memory_utilization_percent': memory_util
            }
            gpu_info['devices'].append(device_info)
            
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info


def get_tensorflow_info():
    """Get TensorFlow GPU detection information."""
    tf_info = {}
    
    try:
        # GPU devices detected by TensorFlow
        physical_devices = tf.config.list_physical_devices('GPU')
        tf_info['gpu_devices'] = len(physical_devices)
        tf_info['gpu_names'] = [device.name for device in physical_devices]
        
        # Memory growth setting
        if physical_devices:
            tf_info['memory_growth'] = []
            for device in physical_devices:
                try:
                    growth = tf.config.experimental.get_memory_growth(device)
                    tf_info['memory_growth'].append(growth)
                except:
                    tf_info['memory_growth'].append(None)
        
        # CUDA version info
        tf_info['built_with_cuda'] = tf.test.is_built_with_cuda()
        
    except Exception as e:
        tf_info['error'] = str(e)
    
    return tf_info


def print_resource_summary():
    """Print a comprehensive resource summary using monospace font in notebook."""
    output_lines = []
    output_lines.append("SYSTEM RESOURCE SUMMARY")
    output_lines.append("=" * 45)
    
    # Basic system info
    hostname = get_hostname()
    output_lines.append(f"Hostname: {hostname}")
    
    # SLURM job info
    slurm_info = get_slurm_info()
    output_lines.append("")
    output_lines.append("Job Allocation:")
    output_lines.append(f"   CPUs: {slurm_info['cpus']}")
    output_lines.append(f"   Memory: {slurm_info['memory_gb']:.1f} GB")
    output_lines.append(f"   Partition: {slurm_info['actual_partition']}")
    output_lines.append(f"   Job ID: {slurm_info['job_id']}")
    output_lines.append(f"   Node list: {slurm_info['node_list']}")
    
    # Physical GPU hardware detection
    gpu_info = get_gpu_info()
    output_lines.append("")
    output_lines.append("Physical GPU Hardware:")
    
    if not gpu_info['available']:
        if 'error' in gpu_info:
            if 'NVML Shared Library Not Found' in gpu_info['error']:
                output_lines.append(f"   No physical GPUs allocated to this job")
            else:
                output_lines.append(f"   Error: {gpu_info['error']}")
        else:
            output_lines.append(f"   No physical GPUs detected")
    else:
        output_lines.append(f"   Physical GPUs available: {gpu_info['count']}")
        for device in gpu_info['devices']:
            output_lines.append(f"   GPU {device['index']}: {device['name']}")
            output_lines.append(f"      Memory: {device['memory_used_gb']:.1f}/"
                              f"{device['memory_total_gb']:.1f} GB "
                              f"({device['memory_free_gb']:.1f} GB free)")
            
            if device['temperature_c'] is not None:
                output_lines.append(f"      Temperature: {device['temperature_c']}°C")
            
            if device['gpu_utilization_percent'] is not None:
                output_lines.append(f"      Utilization: GPU {device['gpu_utilization_percent']}%, "
                                  f"Memory {device['memory_utilization_percent']}%")
    
    # TensorFlow GPU software recognition
    tf_info = get_tensorflow_info()
    output_lines.append("")
    output_lines.append("TensorFlow GPU Recognition:")
    if 'error' in tf_info:
        output_lines.append(f"   Error: {tf_info['error']}")
    else:
        output_lines.append(f"   TensorFlow can access {tf_info.get('gpu_devices', 'N/A')} GPU(s)")
        if tf_info.get('gpu_devices', 0) > 0:
            for i, name in enumerate(tf_info.get('gpu_names', [])):
                growth = tf_info.get('memory_growth', [None]*tf_info.get('gpu_devices', 0))[i] if 'memory_growth' in tf_info else None
                growth_str = f", Memory growth: {growth}" if growth is not None else ""
                output_lines.append(f"      {name}{growth_str}")
        output_lines.append(f"   Built with CUDA support: {tf_info.get('built_with_cuda', 'N/A')}")
    
    output_lines.append("=" * 45)
    
    # Display as monospace, no background
    display(Markdown("<pre>{}</pre>".format('\n'.join(output_lines))))
    
    # Flush stdout to prevent output duplication
    import sys
    sys.stdout.flush()


def get_resource_dict():
    """Return resource information as a dictionary for programmatic use."""
    return {
        'hostname': get_hostname(),
        'slurm': get_slurm_info(),
        'gpu': get_gpu_info(),
        'tensorflow': get_tensorflow_info()
    }
