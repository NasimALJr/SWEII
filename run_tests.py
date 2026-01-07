#!/usr/bin/env python3
"""
Test runner script for the course material recommendation system.
This script runs all tests and provides a summary.
"""

import subprocess
import sys
import os

def run_tests():
    """Run pytest on the test suite"""
    print("Running tests for Course Material Recommendation System...")
    print("=" * 60)

    # Change to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=False, text=True)

        return result.returncode == 0

    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest:")
        print("pip install pytest")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main function"""
    success = run_tests()

    if success:
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("The course material recommendation system is working correctly.")
        return 0
    else:
        print("\n" + "=" * 60)
        print("Some tests failed!")
        print("Please check the test output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())