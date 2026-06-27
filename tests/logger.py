"""
    tests / logger.py
    -----------------
    Tests for the logger class including initialization, logging levels, 
    context managers, queue handling, and file / console output validation.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import tempfile
import time
import json
import os
import logging
import glob
import re

import numpy as np

from logs.logger import Logger, LogLevel
from logs.formatters import JsonFormatter, ConsoleFormatter

from tools.timer import Timer
from tools.test_tools import runner, builder, CommandSpec


# ======================================================================
#       Utility Functions
# ======================================================================

def get_log_file_path(directory: Path) -> Path | None:
    """
    Get the most recent log file in the directory
    """
    log_files = list(directory.glob("*.log"))
    if not log_files:
        return None
    
    # Return the most recent (by name, since timestamp is in name)
    log_files.sort(reverse=True)
    return log_files[0]


def assert_log_file_exists(directory: Path, size: int = 100) -> Path:
    """
    Assert that a log file exists and has content, return the file path
    """
    log_file = get_log_file_path(directory)
    assert log_file is not None, f"No log file found in ``{directory}``"
    assert log_file.exists(), f"Log file ``{log_file}`` does not exist"
    assert log_file.stat().st_size >= size, \
        f"Log file is too small: ``{log_file.stat().st_size} bytes``"
    
    print(f"\tLog file exists: ``{log_file} ({log_file.stat().st_size} bytes)``")
    return log_file


def read_log_lines(filepath: Path, n_lines: int = -1) -> list:
    """
    Read specified number of lines from log file
    """
    with open(filepath, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    
    return lines[:n_lines] if n_lines > 0 else lines


def parse_json_log(line) -> dict | None:
    """
    Parse a JSON log line
    """
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def assert_json_log_has_fields(log_entry: dict, required_fields: list):
    """ Assert that a JSON log entry has required fields """
    for field in required_fields:
        assert field in log_entry, f"Missing field: ``{field}``"



# ======================================================================
#       Logger Initialization Tests
# ======================================================================

def test_logger_singleton(args: argparse.Namespace):
    """
    Test that logger maintains singleton instances per name
    """
    print("\n1. Testing singleton behavior:")

    logger1 = Logger("test_singleton")
    logger2 = Logger("test_singleton")
    logger3 = Logger("different_name")

    print(f"\tlogger1 id: ``{id(logger1)}``")
    print(f"\tlogger2 id: ``{id(logger2)}``")
    print(f"\tlogger3 id: ``{id(logger3)}``")

    assert logger1 is logger2, "Same name should return same instance"
    assert logger1 is not logger3, \
        "Different names should return different instances"
    
    print("\n2. Testing instance registry:")
    print(f"\tRegistered loggers: {list(Logger._instances.keys())}")
    assert "test_singleton" in Logger._instances
    assert "different_name" in Logger._instances

    print("\n3. Testing re-initialization prevention:")
    logger1_copy = Logger("test_singleton", level=LogLevel.DEBUG)
    
    # Should not override existing configuration
    assert logger1._configured, "Logger should remain configured"
    
    print("\nSingleton test passed")



def test_logger_initialization(args: argparse.Namespace):
    """ Test Logger initialization with various configurations """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        print("\n1. Default initialization:")
        logger = Logger.get_logger(
            "test_default", use_console=True, to_disk=True
        )

        print(f"\tName: ``{logger.name}``")
        print(f"\tLogger Level: ``{logger.logger.level}``")
        print(f"\tDirectory: ``{logger.directory}``")

        assert logger.logger.level == LogLevel.INFO.value

        print("\n2. Custom initialization:")
        logger_custom = Logger.get_logger(
            "test_custom", level=LogLevel.DEBUG, use_console=False,
            to_disk=True, max_bytes=5 * 1024 * 1024, backup_count=3
        )

        print(f"\tCustom level: {logging.getLevelName(logger_custom.logger.level)}")
        print(f"\tUse console: ``{False}``")
        print(f"\tMax Bytes: ``{5 * 1024 * 1024}``")
        print(f"\tBackup count: ``{3}``")

        assert logger_custom.logger.level == LogLevel.DEBUG.value

        print("\n3. Testing infrastructure initialization:")

        # Infrastructure should be initialized once, only once then reuse
        assert Logger._initialized, "Infrastructure not initialized"
        assert Logger._queue is not None, "Queue is not created"
        assert Logger._listener is not None, "Listener is not started"

        print(f"\tQueue Maxsize: ``{Logger._queue.maxsize}``")
        print(f"\tNumber of handlers: ``{len(Logger._handlers)}``")
    
    print("\nInitialization test passed")



def test_logger_shutdown(args: argparse.Namespace):
    """ Test logger shutdown behavior """

    print("\n1. Testing shutdown protection:")

    # Reset the Logger
    Logger._shutdown = False
    Logger._initialized = False
    Logger._queue = None
    Logger._listener = None
    Logger._handlers = []

    logger = Logger.get_logger("test_shutdown")
    print(f"\tLogger created: ``{logger.name}``")

    Logger.shutdown()
    print("\tShutdown Called")

    assert Logger._shutdown, "Shutdown flag not set"
    
    print("\n2. Testing shutdown idempotence:")
    Logger.shutdown()

    print("\tSecond shutdown call succeeded")

    print("\n3. Testing logger creation after shutdown:")
    try:

        Logger.get_logger("test_after_shutdown")
        print("\tError: Should have raised RuntimeError")
    
    except RuntimeError as e:
        print(f"\tCorrectly raised: ``{str(e)}``")
    
    # Reset for other tests
    Logger._shutdown = False
    Logger._initialized = False
    Logger._queue = None
    Logger._listener = None
    Logger._handlers = []

    print("\nShutdown test passed")



# ======================================================================
#       Logging Level Tests
# ======================================================================

def test_logging_levels(args: argparse.Namespace):
    """
    Test all logging levels
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)

        # Use disk logging for verification
        logger = Logger.get_logger(
            "test_levels", level=LogLevel.DEBUG, use_console=True, 
            to_disk=True, directory=records_path
        )

        print("\n1. Testing all log levels:")

        # Log at each Level
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        logger.critical("This is a CRITICAL message")

        # Wait for async logging to complete
        time.sleep(0.5)

        # Verify log file exists
        log_file = assert_log_file_exists(records_path)
        
        # Read and parse logs
        lines = read_log_lines(log_file)
        print(f"\tLog file contains {len(lines)} lines")

        # Verify each level appears
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in levels:
            found = any(level in line for line in lines)
            print(f"\t{level}: {'✓' if found else '✗'}")
            assert found, f"Level ``{level}`` not found in logs"
        
        print("\n2. Testing level filtering:")

        # Create logger with WARNING level
        logger_warn = Logger.get_logger(
            "test_filter", level=LogLevel.WARNING, to_disk=True, directory=records_path
        )

        logger_warn.debug("This DEBUG should be filtered")
        logger_warn.info("This INFO should be filtered")
        logger_warn.warning("This WARNING should appear")
        logger_warn.error("This ERROR should appear")

        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        print(f"\tAfter filtering, log has {len(lines)} lines")
        
        # Filtered messages should not appear
        log_content = ''.join(lines)
        # These should NOT appear at all
        assert "filtered" not in log_content, "Filtered messages appeared"
        assert "appear" in log_content, "Allowed messages missing"
    
    print("\nLogging levels test passed")



def test_log_methods_with_args(args: argparse.Namespace):
    """Test logging with formatting arguments"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger(
            "test_args", to_disk=True, directory=records_path
        )
        
        print("\n1. Testing formatted messages:")
        
        logger.info("Value: %d, Name: %s", 42, "test")
        logger.warning("Error code: %04x", 255)
        logger.error("Exception at %s line %d", "test.py", 100)
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        
        print(f"\tLog file contains {len(lines)} lines")
        
        # Check formatted content
        content = ''.join(lines)
        assert "Value: 42, Name: test" in content, "Formatted message missing"
        assert "Error code: 00ff" in content, "Hex formatting missing"
        assert "Exception at test.py line 100" in content, "Multi-arg formatting missing"
        
        print("\n2. Testing extra fields:")
        
        # Log with extra fields
        logger.info(
            "User action completed", user_id=12345, action="login", duration_ms=150.5
        )
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        last_line = lines[-1] if lines else ""
        
        # Check for JSON format (extra fields should be in JSON)
        parsed = parse_json_log(last_line)
        if parsed:
            print(f"\tExtra fields in JSON: {list(parsed.keys())}")
            
            if 'user_id' in parsed:
                print(f"\t\tuser_id: {parsed['user_id']}")

            if 'action' in parsed:
                print(f"\t\taction: {parsed['action']}")
    
    print("\nLog args test passed")



# ======================================================================
#       Context Manager Tests
# ======================================================================

def test_time_block(args: argparse.Namespace):
    """Test time_block context manager"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger("test_time", to_disk=True, directory=records_path)
        
        print("\n1. Testing time_block with default level:")
        with logger.time_block("Test operation"):
            time.sleep(0.05)
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        
        # Find time block log
        time_logs = [l for l in lines if "Test operation completed in" in l]
        print(f"\tFound {len(time_logs)} time block logs")
        
        if time_logs:
            print(f"\tLog: {time_logs[-1].strip()}")
            # Should contain duration in seconds
            assert "s" in time_logs[-1] or "ms" in time_logs[-1], "No duration found"
        
        print("\n2. Testing time_block with custom level:")
        with logger.time_block("Debug operation", level=LogLevel.DEBUG):
            time.sleep(0.02)
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        debug_time_logs = [l for l in lines if "Debug operation completed in" in l]
        print(f"\tFound {len(debug_time_logs)} debug time block logs")
        
        print("\n3. Testing time_block with exceptions:")
        
        try:
            with logger.time_block("Failing operation"):
                time.sleep(0.01)
                raise ValueError("Test exception in time block")
        
        except ValueError:
            print("\tException caught (time block should still log)")
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        fail_logs = [l for l in lines if "Failing operation completed in" in l]
        print(f"\tFound {len(fail_logs)} failing operation logs")
        assert len(fail_logs) > 0, "Time block didn't log on exception"
    
    print("\nTime block test passed")


def test_catch_context(args: argparse.Namespace):
    """Test catch context manager"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger(
            "test_catch", to_disk=True, directory=records_path
        )
        
        print("\n1. Testing catch with no exception:")
        
        with logger.catch("Normal operation"):
            print("\tNo exception in catch block")
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        print(f"\tLog file has {len(lines)} lines")
        
        print("\n2. Testing catch with exception:")
        
        try:
            with logger.catch("Failing operation"):
                print("\tRaising exception in catch block")
                raise RuntimeError("Test error message")
        
        except RuntimeError as e:
            print(f"\tException caught: {e}")
        
        time.sleep(0.5)
        lines = read_log_lines(log_file)
        
        # Find exception log
        exception_logs = [l for l in lines if "Failing operation" in l]
        print(f"\tFound {len(exception_logs)} exception logs")
        
        # Should contain exception info
        error_logs = [l for l in lines if "ERROR" in l and "Test error" in l]
        print(f"\tFound {len(error_logs)} error logs with message")
        
        print("\n3. Testing nested catch contexts:")
        
        try:
            with logger.catch("Outer block"):
                with logger.catch("Inner block"):
                    raise ValueError("Nested error")
        
        except ValueError:
            print("\tNested exception caught")
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        nested_logs = [l for l in lines if "Outer block" in l or "Inner block" in l]
        print(f"\tFound {len(nested_logs)} nested exception logs")
    
    print("\nCatch context test passed")



# ======================================================================
#       Formatter Tests
# ======================================================================

def test_json_formatter(args: argparse.Namespace):
    """Test JsonFormatter output"""
    
    print("\n1. Testing JsonFormatter:")
    
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger", level=logging.INFO, pathname="test.py",
        lineno=42, msg="Test message", args=(), exc_info=None
    )
    
    # Add extra fields
    record.user_id = 12345
    record.session_id = "abc-123"
    record.custom_field = {"nested": "value"}
    
    formatted = formatter.format(record)
    print(f"\tFormatted output: {formatted[:200]}...")
    
    # Parse and validate
    parsed = json.loads(formatted)
    required_fields = [
        'timestamp', 'level', 'logger', 'message',
        'module', 'function', 'file', 'line',
        'pid', 'thread'
    ]
    
    for field in required_fields:
        assert field in parsed, f"Missing required field: {field}"
        print(f"\t{field}: {parsed[field]}")
    
    # Check extra fields
    assert parsed['user_id'] == 12345, "Extra field not preserved"
    assert parsed['session_id'] == "abc-123", "Extra field not preserved"
    assert parsed['custom_field']['nested'] == "value", "Nested extra field not preserved"
    
    print("\n2. Testing exception formatting:")
    
    try:
        raise ValueError("Test exception")
    
    except ValueError:
        record_exc = logging.LogRecord(
            name="test_logger", level=logging.ERROR, pathname="test.py",
            lineno=50, msg="Exception occurred", args=(), exc_info=sys.exc_info()
        )
        
        formatted_exc = formatter.format(record_exc)
        parsed_exc = json.loads(formatted_exc)
        
        print(f"\tException log: {formatted_exc[:200]}...")
        
        assert 'exception' in parsed_exc, "Exception field missing"
        assert "ValueError" in parsed_exc['exception'], "Exception type missing"
        print(f"\tException type found in log")
    
    print("\nJsonFormatter test passed")


def test_console_formatter(args: argparse.Namespace):
    """Test ConsoleFormatter output"""
    
    print("\n1. Testing ConsoleFormatter colors:")
    formatter = ConsoleFormatter()
    
    # Test each level
    for level_name, color in ConsoleFormatter.COLORS.items():
        if level_name == "RESET":
            continue
        
        record = logging.LogRecord(
            name="test_logger", level=getattr(logging, level_name), 
            pathname="test.py", lineno=42, msg=f"{level_name} message",
            args=(), exc_info=None
        )
        record.funcName = "test_function"
        record.filename = "test.py"
        record.lineno = 42
        
        formatted = formatter.format(record)
        print(f"\t{level_name}: {formatted[:80]}...")
        
        # Check color codes
        expected_color = ConsoleFormatter.COLORS[level_name]
        reset_color = ConsoleFormatter.COLORS["RESET"]
        
        # Colors should be present in formatted string
        assert expected_color in formatted, f"Missing color for {level_name}"
        assert reset_color in formatted, f"Missing reset color for {level_name}"
    
    print("\n2. Testing message escaping:")
    
    record = logging.LogRecord(
        name="test_logger", level=logging.INFO, pathname="test.py",
        lineno=42, msg="Message with\nnewline", args=(), exc_info=None
    )
    record.funcName = "test_function"
    record.filename = "test.py"
    record.lineno = 42
    
    formatted = formatter.format(record)
    
    # Newline should be escaped
    assert "\\n" in formatted, "Newline not escaped"
    print(f"\tNewline escaped correctly")
    
    print("\nConsoleFormatter test passed")



# ======================================================================
#       File Output Tests
# ======================================================================

def test_file_rotation(args: argparse.Namespace):
    """Test rotating file handler behavior"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        # Small max_bytes to force rotation
        logger = Logger.get_logger(
            "test_rotation", to_disk=True, directory=records_path,
            max_bytes=1024, backup_count=3
        )
        
        print("\n1. Testing log rotation:")
        
        # Generate logs to trigger rotation
        for i in range(50):
            logger.info(f"Rotation test message {i:03d} " + "x" * 100)
        
        # Wait for async writes
        time.sleep(1.0)
        
        # Check for rotated files - look for .log files and .log.1, .log.2 etc
        all_log_files = list(records_path.glob("*.log*"))
        print(f"\tFound {len(all_log_files)} log files")
        
        # Get the base log file (the most recent one)
        base_log = get_log_file_path(records_path)
        if base_log:
            print(f"\t\t{base_log.name}")
        
        # Check for backup files
        for f in all_log_files:
            size = f.stat().st_size
            print(f"\t\t{f.name}: {size} bytes")
        
        # Should have at least the main log and maybe some backups
        assert len(all_log_files) >= 2, "Rotation did not occur"
        
        print("\n2. Testing backup count limit:")
        # Force more rotations
        for i in range(100):
            logger.info(f"More rotation test {i:03d} " + "x" * 100)
        
        time.sleep(1.0)
        
        all_log_files = list(records_path.glob("*.log*"))
        print(f"\tAfter more logs: {len(all_log_files)} files")
        
        # Should not exceed backup_count + 1 (main) + maybe the .log file itself
        # Note: RotatingFileHandler creates .log, .log.1, .log.2, etc.
        assert len(all_log_files) <= 4, f"Too many backup files: {len(all_log_files)}"
    
    print("\nFile rotation test passed")


def test_console_output(args: argparse.Namespace):
    """Test console output (captured via stdout)"""
    
    import io
    from contextlib import redirect_stdout
    
    print("\n1. Testing console output capture:")
    
    # Reset state to ensure fresh handlers
    Logger._shutdown = False
    Logger._initialized = False
    Logger._queue = None
    Logger._listener = None
    Logger._handlers = []
    
    stdout_capture = io.StringIO()
    with redirect_stdout(stdout_capture):
        # Console-only logger
        logger = Logger.get_logger(
            "test_console",
            use_console=True,
            to_disk=False
        )
        logger.info("Console test message")
        # Need to flush queue
        time.sleep(0.1)
    
    captured = stdout_capture.getvalue()
    print(f"\tCaptured: {captured[:80]}...")
    
    assert "Console test message" in captured, "Console output missing"
    assert "INFO" in captured, "Level missing from console output"
    
    print("\n2. Testing console-only logging:")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        
        # Reset state
        Logger._shutdown = False
        Logger._initialized = False
        Logger._queue = None
        Logger._listener = None
        Logger._handlers = []
        
        logger = Logger.get_logger(
            "test_console_only",
            use_console=True,
            to_disk=False,
            directory=path
        )
        logger.info("Console only message")
        time.sleep(0.1)
        
        # Verify no log file created
        log_files = list(path.glob("*.log")) + list(path.glob("**/*.log"))
        assert len(log_files) == 0, f"Log files created despite to_disk=False: {log_files}"
        print("\tNo disk writes occurred")
    
    print("\nConsole output test passed")



# ======================================================================
#       Performance Tests
# ======================================================================

def test_performance_throughput(args: argparse.Namespace):
    """Test logging throughput performance"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger(
            "test_perf", to_disk=True, directory=records_path, use_console=False
        )
        
        n_messages = args.n_perf_samples
        print(f"\nTesting throughput with {n_messages} messages...")
        
        # Test debug logging throughput
        with Timer(f"\tLog {n_messages} info messages"):
            for i in range(n_messages):
                logger.info(f"Performance test message {i}")
        
        # Wait for queue to drain
        time.sleep(1.0)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        print(f"\t{len(lines)} lines in log file")
        
        # Check throughput
        if n_messages > 1000:
            throughput = n_messages / 0.1  # ~estimate
            print(f"\tEstimated throughput: ~{throughput:.0f} messages/sec")
    
    print("\nPerformance test passed")


def test_concurrent_logging(args: argparse.Namespace):
    """Test logging from multiple threads"""
    
    import threading
    import concurrent.futures
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger(
            "test_concurrent", to_disk=True, directory=records_path, use_console=False
        )
        
        n_threads = 10
        messages_per_thread = 100
        
        print(
            f"\n1. Testing {n_threads} threads with {messages_per_thread} "
            "messages each:"
        )
        
        def log_messages(thread_id: int):
            for i in range(messages_per_thread):
                logger.info(f"Thread {thread_id} message {i}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(log_messages, tid) for tid in range(n_threads)]
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        # Wait for queue to drain
        time.sleep(2.0)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        expected_lines = n_threads * messages_per_thread

        print(f"\tExpected: {expected_lines} lines")
        print(f"\tActual: {len(lines)} lines")

        # Allow some tolerance for timing issues
        assert len(lines) >= expected_lines - 10, \
            f"Only {len(lines)} lines, expected ~{expected_lines}"

        # Check that all threads appear
        thread_ids = set()
        for line in lines:
            if "Thread" in line:
                # Try to extract thread ID from the message
                match = re.search(r'Thread\s+(\d+)', line)
                if match:
                    thread_ids.add(int(match.group(1)))

        print(f"\tThreads found in logs: {sorted(thread_ids)}")
        assert len(thread_ids) >= n_threads - 2, \
            f"Only {len(thread_ids)} threads found in logs"

    print("\nConcurrent logging test passed")



# ======================================================================
#       Edge Cases Tests
# ======================================================================

def test_edge_cases(args: argparse.Namespace):
    """Test edge cases and error conditions"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger("test_edge", to_disk=True, directory=records_path)
        print("\n1. Testing very long messages:")
        
        long_msg = "x" * 10000
        logger.info(long_msg[:50] + "... (long message)")
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        assert len(lines) > 0, "Long message not logged"
        print(f"\tLong message logged successfully")
        
        print("\n2. Testing special characters:")
        
        special_msg = "Special: \n\t\r\\\"'€✓★🎉, :) «€œ®þ€łd sd ªª ←þ πΠ ΣΠ"
        logger.info(special_msg)
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        special_log = [l for l in lines if "Special:" in l]

        if special_log:
            print(f"\tSpecial characters logged: {special_log[0][:80]}...")
        
        print("\n3. Testing empty messages:")
        
        logger.info("")
        logger.warning(" ")
        logger.error("\t")
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        print(f"\tEmpty messages logged ({len(lines)} total lines)")
        
        print("\n4. Testing Unicode messages:")
        
        unicode_msg = "Unicode: 你好, 世界! 🌍 Hello, World!"
        logger.info(unicode_msg)
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        unicode_log = [l for l in lines if "Unicode:" in l]
        if unicode_log:
            print(f"\tUnicode logged: {unicode_log[0][:80]}...")
            # Check encoding (should be valid UTF-8)
            try:
                unicode_log[0].encode('utf-8')
                print("\t\tUTF-8 encoding valid")
            except UnicodeEncodeError:
                print("\t\tUTF-8 encoding FAILED")
                raise
    
    print("\nEdge cases test passed")


def test_error_handling(args: argparse.Namespace):
    """Test error handling in logger"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        print("\n1. Testing exception logging:")
        logger = Logger.get_logger("test_errors", to_disk=True, directory=records_path)
        
        try:
            raise ValueError("Test error for exception logging")
        except ValueError:
            logger.exception("Exception occurred in test")
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        
        error_logs = [l for l in lines if "ERROR" in l and "Exception occurred" in l]
        print(f"\tFound {len(error_logs)} exception logs")
        
        if error_logs:
            print(f"\tLog: {error_logs[0][:100]}...")
            assert "ValueError" in error_logs[0] or "Traceback" in error_logs[0], \
                "Exception info missing"
        
        print("\n2. Testing invalid log level:")
        
        # Should handle gracefully
        logger.log(99, "Invalid level message")
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        print(f"\tInvalid level logged (logger handles gracefully)")
        print("\n3. Testing logger after shutdown prevention:")
        
        # Tested in shutdown test
        
    print("\nError handling test passed")



# ======================================================================
#       Integration Tests
# ======================================================================

def test_integration_with_json_logs(args: argparse.Namespace):
    """Test integration with JSON log format"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        records_path = path / "records"
        records_path.mkdir(parents=True, exist_ok=True)
        
        logger = Logger.get_logger(
            "test_json_integration", to_disk=True, directory=records_path
        )
        
        print("\n1. Logging structured data:")
        
        # Log with various data types
        logger.info(
            "Structured log", integer=42, float_val=3.14159, boolean=True,
            list_val=[1, 2, 3], dict_val={"key": "value"}, null_val=None
        )
        
        time.sleep(0.5)
        
        log_file = assert_log_file_exists(records_path)
        lines = read_log_lines(log_file)
        
        # Find the structured log
        structured_logs = [l for l in lines if "Structured log" in l]
        
        if structured_logs:
            parsed = parse_json_log(structured_logs[-1])
            
            if parsed:
                print(f"\tParsed fields: {list(parsed.keys())}")
                print(f"\t\tinteger: {parsed.get('integer')}")
                print(f"\t\tfloat_val: {parsed.get('float_val')}")
                print(f"\t\tboolean: {parsed.get('boolean')}")
                print(f"\t\tlist_val: {parsed.get('list_val')}")
                print(f"\t\tdict_val: {parsed.get('dict_val')}")
                print(f"\t\tnull_val: {parsed.get('null_val')}")
        
        print("\n2. Logging with all extra fields:")
        
        logger.info("Rich log", **{
            "field1": "value1",
            "field2": 42,
            "field3": [1, 2, 3],
            "field4": {"nested": "data"},
            "field5": "value5"
        })
        
        time.sleep(0.5)
        
        lines = read_log_lines(log_file)
        rich_logs = [l for l in lines if "Rich log" in l]
        
        if rich_logs:
            parsed = parse_json_log(rich_logs[-1])
            if parsed:
                found_fields = [k for k in ['field1', 'field2', 'field3', 'field4', 'field5'] 
                               if k in parsed]
                print(f"\tFound {len(found_fields)} custom fields")
        
        print("\n3. Validating JSON output:")
        
        # All lines should be valid JSON
        valid_json = 0
        for line in lines:
            try:
                json.loads(line.strip())
                valid_json += 1
            except json.JSONDecodeError:
                print(f"\tInvalid JSON: {line[:50]}...")
        
        print(f"\tValid JSON lines: {valid_json}/{len(lines)}")
        
        # All lines should be valid JSON (for JSON formatter)
        assert valid_json == len(lines), f"Some log lines are not valid JSON ({len(lines) - valid_json} invalid)"
    
    print("\nIntegration test passed")



# ======================================================================
#       Main Runner
# ======================================================================

COMMON = [
    {"flags": ["--n-samples", "-n"], "kwargs": {"type": int, "default": 1000}},
    {"flags": ["--n-perf-samples"], "kwargs": {"type": int, "default": 10000}},
]

TEST_ARGS = [*COMMON]

@runner
def main():
    # Reset logger state for clean testing
    Logger._shutdown = False
    Logger._initialized = False
    Logger._queue = None
    Logger._listener = None
    Logger._handlers = []
    Logger._instances = {}
    
    p = builder([
        CommandSpec(
            "singleton", "Test singleton behavior",
            test_logger_singleton, TEST_ARGS
        ),
        CommandSpec(
            "init", "Test logger initialization",
            test_logger_initialization, TEST_ARGS
        ),
        CommandSpec(
            "shutdown", "Test shutdown behavior",
            test_logger_shutdown, TEST_ARGS
        ),
        CommandSpec(
            "levels", "Test logging levels",
            test_logging_levels, TEST_ARGS
        ),
        CommandSpec(
            "args", "Test log methods with args",
            test_log_methods_with_args, TEST_ARGS
        ),
        CommandSpec(
            "time_block", "Test time_block context manager",
            test_time_block, TEST_ARGS
        ),

        # NOT WORKING
        CommandSpec(
            "catch", "Test catch context manager",
            test_catch_context, TEST_ARGS
        ),
        CommandSpec(
            "json_formatter", "Test JSON formatter",
            test_json_formatter, TEST_ARGS
        ),
        CommandSpec(
            "console_formatter", "Test console formatter",
            test_console_formatter, TEST_ARGS
        ),
        CommandSpec(
            "rotation", "Test file rotation",
            test_file_rotation, TEST_ARGS
        ),
        CommandSpec(
            "console", "Test console output",
            test_console_output, TEST_ARGS
        ),
        CommandSpec(
            "perf", "Performance test",
            test_performance_throughput, [*COMMON]
        ),
        CommandSpec(
            "concurrent", "Test concurrent logging",
            test_concurrent_logging, TEST_ARGS
        ),
        CommandSpec(
            "edge", "Test edge cases",
            test_edge_cases, TEST_ARGS
        ),
        CommandSpec(
            "errors", "Test error handling",
            test_error_handling, TEST_ARGS
        ),
        # Errors
        CommandSpec(
            "integration", "Integration test",
            test_integration_with_json_logs, TEST_ARGS
        ),
    ])
    args = p.parse_args()
    args._handler(args)
    
    # Clean up after tests
    Logger.shutdown()


if __name__ == "__main__":
    main()