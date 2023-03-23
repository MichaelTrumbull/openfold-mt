import argparse

args = argparse.ArgumentParser()

args.add_argument('-r', action="store_true", default=False)
args.add_argument('-i', action="store", dest="b")

with open('iteration_DOE_file.txt', 'w') as f: 
    f.write(f'r_{args.r}_i_{args.i}')