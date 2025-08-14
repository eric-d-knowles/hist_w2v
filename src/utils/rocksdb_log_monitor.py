import time
import threading
import logging

def monitor_rocksdb_log(log_path, poll_interval=10, logger=None):
    """
    Monitor RocksDB log file for compaction events and log them.
    Args:
        log_path: Path to the RocksDB LOG file.
        poll_interval: Seconds between polling for new log lines.
        logger: Optional Python logger. If None, uses print().
    """
    compaction_active = False
    try:
        while True:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)  # Go to end of file
                filesize = f.tell()
                blocksize = min(65536, filesize)
                f.seek(-blocksize, 2) if filesize > blocksize else f.seek(0)
                lines = f.read().decode(errors='ignore').splitlines()[::-1]  # Reverse order
                for line in lines:
                    if "compaction_finished" in line:
                        if compaction_active:
                            msg = f"RocksDB compaction finished: {line.strip()}"
                            if logger:
                                logger.info(msg)
                            else:
                                print(msg)
                        compaction_active = False
                        break
                    elif "compaction_started" in line:
                        if not compaction_active:
                            msg = f"RocksDB compaction detected: {line.strip()}"
                            if logger:
                                logger.info(msg)
                            else:
                                print(msg)
                        compaction_active = True
                        break
            time.sleep(poll_interval)
    except Exception as e:
        if logger:
            logger.error(f"Error monitoring RocksDB log: {e}")
        else:
            print(f"Error monitoring RocksDB log: {e}")

def start_rocksdb_log_monitor(log_path, poll_interval=10, logger=None):
    """
    Start the RocksDB log monitor in a background thread.
    Returns the thread object.
    """
    t = threading.Thread(
        target=monitor_rocksdb_log,
        args=(log_path, poll_interval, logger),
        daemon=True
    )
    t.start()
    return t
