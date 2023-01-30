import os
protiens_dir = "/scratch/00946/zzhang/data/openfold/cameo/alignments/"
output_file = open("Neff_for_vdata.txt", "w")
for protien in os.listdir(protiens_dir):
    hits_file = protiens_dir + protien + "/" + "pdb70_hits.hhr"
    with open(hits_file) as f:
        lines = f.readlines()
    output_file.write(protien + " " + lines[3])
output_file.close()