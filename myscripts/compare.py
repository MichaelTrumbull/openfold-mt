from Bio.PDB import *
import nglview as nv
import ipywidgets

cif_parser = MMCIFParser()
structure = cif_parser.get_structure("4ja4", "target_struc/4ja4.cif")
view = nv.show_biopython(structure)