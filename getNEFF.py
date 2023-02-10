# get values from hmm files. NOTE: THE EFFECTIVE NUMBER OF SEQUENCES IS NOT HERE. SEE HMMSTATS
'''
import os
protiens_dir = "hmmfiles/" #"/scratch/00946/zzhang/data/openfold/cameo/alignments/"
output_file = open("eff_nseq_for_vdata.txt", "w")
for protien in os.listdir(protiens_dir):
    hmm_file = protiens_dir + protien
    with open(hmm_file) as f:
        lines = f.readlines()
    output_file.write(protien + " " + lines[10])

output_file.close()
'''

########################
#get values from hmmstats files...
import os
protiens_dir = "hmmstats/" 
output_file = open("hmmstat.txt", "w")
for protien in os.listdir(protiens_dir):
    hmm_file = protiens_dir + protien
    with open(hmm_file) as f:
        lines = f.readlines()
    output_file.write(protien + "  " + lines[8])

output_file.close()

# get value from hmmfile. specifically looking for: (even though i don't know what they are)
#STATS LOCAL MSV    ## LINE [13]
#STATS LOCAL VITERBI ## LINE [14]
#STATS LOCAL FORWARD ## LINE [15]
import os
protiens_dir = "hmmfiles/" 
output_file = open("FORWARD_from_hmmfile.txt", "w")
for protien in os.listdir(protiens_dir):
    hmm_file = protiens_dir + protien
    with open(hmm_file) as f:
        lines = f.readlines()
    try:
        output_file.write(protien + "  " + lines[15])
    except:
        print(protien, 'failed')

output_file.close()

import os
protiens_dir = "hmmfiles/" 
output_file = open("EFFN_from_hmmfile.txt", "w")
for protien in os.listdir(protiens_dir):
    hmm_file = protiens_dir + protien
    with open(hmm_file) as f:
        lines = f.readlines()
    try:
        output_file.write(protien + "  " + lines[11])
    except:
        print(protien, 'failed')

output_file.close()


# I'm putting this here to see if the protein files in the fasta dir are the same as in the align dir
import os
fasta_dir = "/work/09123/mjt2211/ls6/openfold-mt/cameo_dir/fasta_dir"
align_dir = "/scratch/09120/sk844/validation_set_cameo/alignments"
hold_align = []
for name in os.listdir(align_dir):
    hold_align.append(name)

for name in os.listdir(fasta_dir):
    if not name[0:6] in hold_align:
            print(name)

hold_fasta = []
for name in os.listdir(fasta_dir):
    hold_fasta.append(name[0:6])

for name in os.listdir(align_dir):
    if not name in hold_fasta:
            print(name)
