import argparse
import time

def parser():
    """
    define command line arguments
    """
    parser = argparse.ArgumentParser(description='main.py command line argument parser')
    parser.add_argument('-test','--test', action="store_true", help="test flag")
    args = parser.parse_args()
    return args


def main():
    """
    main function
    """
    args = parser()

if __name__ == "__main__":
    t0 = time.time()
    main()
    run_time = time.time() - t0
    print("Run time in %.3f s" % run_time)
