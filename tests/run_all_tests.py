"""
City OS Test Runner
Runs all diagnostic tests in the proper order
"""
import subprocess
import sys
import os
import time

def run_command(cmd: str, description: str):
    """Run a shell command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ PASS: {description}")
            print(result.stdout)
        else:
            print(f"✗ FAIL: {description}")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    """Run all tests in sequence."""
    print("="*60)
    print("CITY OS DIAGNOSTIC TEST SUITE")
    print("="*60)
    
    tests = [
        ("python tests/buffer_monitor.py", "Frame Buffer Monitor"),
        ("python tests/ai_pipeline_test.py", "AI Pipeline Sanity Test"),
        ("python tests/infrastructure_healthcheck.py", "Infrastructure Health Check"),
        ("python tests/stream_simulator.py &", "RTSP Stream Simulator (background)"),
    ]
    
    results = []
    for cmd, desc in tests:
        if "&" in cmd:
            print(f"\nINFO: {desc} started in background (Ctrl+C to stop)")
            subprocess.Popen(cmd.replace("&", ""), shell=True)
        else:
            results.append(run_command(cmd, desc))
    
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests PASSED!")
    else:
        print(f"✗ {total - passed} tests FAILED - Review output above")
    
    print("\nTo run WebSocket stress test separately:")
    print("  python tests/websocket_stress_tester.py")
    print("\nTo run resource monitor (Linux/macOS):")
    print("  bash tests/resource_monitor.sh")

if __name__ == "__main__":
    main()