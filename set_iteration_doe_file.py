import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-r')
parser.add_argument('-i')
parser.add_argument('-rep')
args = parser.parse_args()

with open('iteration_DOE_file.txt', 'w') as f: 
    f.write(f'r_{args.r}_i_{args.i}_rep_{args.rep}')