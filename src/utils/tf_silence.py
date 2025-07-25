"""
Simple TensorFlow import with silencing.

Just import TensorFlow quietly without all the noise.
"""

import os
import sys
import warnings
from contextlib import contextmanager


@contextmanager
def silence_tensorflow():
    """
    Context manager to completely silence TensorFlow output.
    
    This redirects stderr to /dev/null during TensorFlow operations,
    which is the only way to silence C++ core messages.
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = open('/dev/null', 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


def setup_tf_environment(force_cpu=False):
    """
    Set up TensorFlow environment variables for current and child processes.
    
    This ensures that multiprocessing workers inherit the quiet settings.
    """
    # Suppress ALL messages except FATAL errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # No oneDNN messages
    os.environ['TF_SUPPRESS_LOGS'] = '1'  # Additional suppression
    
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def import_tf_quietly(force_cpu=False):
    """
    Import TensorFlow with minimal noise.
    
    Simple, reliable function that works every time.
    
    Parameters
    ----------
    force_cpu : bool, optional
        If True, force CPU-only mode. Default False.
        
    Returns
    -------
    tf
        TensorFlow module
    """
    # Set up environment for current and future processes
    setup_tf_environment(force_cpu=force_cpu)
    
    # Suppress Python warnings during import
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # Temporarily redirect stderr during import
        old_stderr = sys.stderr
        try:
            sys.stderr = open('/dev/null', 'w')
            import tensorflow as tf
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
    
    # Set logger to FATAL after import
    tf.get_logger().setLevel('FATAL')
    
    # Set a random seed to avoid runtime errors
    tf.random.set_seed(42)
    
    return tf