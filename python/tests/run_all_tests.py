"""
Run all OCTproEngine Python tests
"""

import sys
import os
import subprocess
import time
import glob

def run_test_with_output(test_file):
    """Run a test and show output in real-time"""
    try:
        # Use shell=False and pass args as list for better compatibility
        process = subprocess.Popen(
            [sys.executable, test_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line and print immediately
        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        return return_code, ''.join(output_lines)
    
    except Exception as e:
        print(f"ERROR running test: {e}")
        import traceback
        traceback.print_exc()
        return 1, str(e)

def main():
    print("=" * 60)
    print("OCTproEngine Python Test Suite")
    print("=" * 60)
    print()
    
    # Auto-discover all test_*.py files
    test_files = sorted(glob.glob('test_*.py'))
    
    if not test_files:
        print("ERROR: No test files found (test_*.py)")
        return 1
    
    print(f"Found {len(test_files)} test(s):")
    for test in test_files:
        print(f"  - {test}")
    print()
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    failed_tests = []
    
    start_time = time.time()
    
    for test in test_files:
        if not os.path.exists(test):
            print(f"WARNING: Test file not found: {test}")
            continue
        
        print(f"Running {test}...")
        print("-" * 60)
        
        test_start = time.time()
        return_code, output = run_test_with_output(test)
        test_duration = time.time() - test_start
        
        print("-" * 60)
        
        if return_code == 0:
            print(f"{test} PASSED ({test_duration:.2f}s)")
            passed += 1
        else:
            print(f"{test} FAILED (exit code: {return_code}, {test_duration:.2f}s)")
            failed += 1
            failed_tests.append(test)
        
        print()
    
    total_duration = time.time() - start_time
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(test_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.2f}s")
    
    if failed_tests:
        print()
        print("Failed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
