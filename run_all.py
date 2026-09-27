"""
Master Entry Point for Assignment 3 Code & Experiments
Author: Tom Des Heath (24888923)
"""
import sys
import subprocess

def main():
    cmd = [sys.executable, "run_experiments.py"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
