import os 

for f in os.listdir("complexes"):
    path = os.path.join("complexes", f)
    with open(path, "r") as fp:
        fasta = fp.read()

    fasta = fasta.split('\n')
    basename = os.path.splitext(f)[0]
    counter = 1
    for tag, seq in zip(fasta[::2], fasta[1::2]):
        new_filename = f"{basename}_{counter}.fasta"
        with open(os.path.join("complex_seqs", new_filename), "w") as fp:
            fp.write(f"{tag}\n{seq}")
        counter += 1
