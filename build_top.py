from utils import *
import argparse


parser = argparse.ArgumentParser(
    description="Structure builder for pi-conjugated peptides"
)
parser.add_argument(
    "--seq",
    type=str,
    default=None,
    metavar="seq",
    help="Structure to create topology for (e.g. DFAGG-4T)",
)
args = parser.parse_args()

if args.seq is None:
    raise ValueError(
        "Input sequence required. (e.g. python build_top.py --seq DFAGG-4T)"
    )

wing, core = args.seq.split("-")
wing_size = len(wing)

print(f"Peptide wing = {wing}")
print(f"Core = {core}")

# import pdb
#
# pdb.set_trace()
wing_trjs = [get_AA(letter) for letter in wing]
wing_Gs = [get_AA(letter, return_graph=True) for letter in wing]
template_trj = get_template(core, wing_size)
template_G = get_template(core, wing_size, return_graph=True)
