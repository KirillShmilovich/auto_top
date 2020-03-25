from utils import *
import argparse
import os


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

wing_trjs = [get_AA(letter) for letter in wing]
wing_Gs = [get_AA(letter, return_graph=True) for letter in wing]
template_trj = get_template(core, wing_size)
template_G = get_template(core, wing_size, return_graph=True)

L_directions = [
    get_SC_direction(template_trj, template_G, f"L{i}") for i in range(wing_size)
]
R_directions = [
    get_SC_direction(template_trj, template_G, f"R{i}") for i in range(wing_size)
]

L_SC_rotations = [
    get_SC_rotation(AA_trj, AA_G, target_ev)
    for AA_trj, AA_G, target_ev in zip(wing_trjs, wing_Gs, L_directions)
]
R_SC_rotations = [
    get_SC_rotation(AA_trj, AA_G, target_ev)
    for AA_trj, AA_G, target_ev in zip(wing_trjs, wing_Gs, R_directions)
]

wing_SC_trjs = [get_SC(AA_trj, AA_G) for AA_trj, AA_G in zip(wing_trjs, wing_Gs)]
wing_SC_Gs = [
    get_SC_G(SC_trj, AA_trj, AA_G)
    for SC_trj, AA_trj, AA_G in zip(wing_SC_trjs, wing_trjs, wing_Gs)
]

L_template_anchor = [
    get_template_position(template_trj, template_G, f"HL{i}") for i in range(wing_size)
]
R_template_anchor = [
    get_template_position(template_trj, template_G, f"HR{i}") for i in range(wing_size)
]

L_SC_shifted_trjs = [
    shift_SC(SC_trj, SC_G, rotation, anchor)
    for SC_trj, SC_G, rotation, anchor in zip(
        wing_SC_trjs, wing_SC_Gs, L_SC_rotations, L_template_anchor
    )
]

R_SC_shifted_trjs = [
    shift_SC(SC_trj, SC_G, rotation, anchor)
    for SC_trj, SC_G, rotation, anchor in zip(
        wing_SC_trjs, wing_SC_Gs, R_SC_rotations, R_template_anchor
    )
]

template_removed_trj = remove_anchors(template_trj, template_G)

if not os.path.isdir("./temp/"):
    os.makedirs("./temp/")

template_removed_trj.save_xtz("./temp/template_removed.xyz")

for i, (L_SC, R_SC) in enumerate(zip(L_SC_shifted_trjs, R_SC_shifted_trjs)):
    L_SC.save_xyz(f"./temp/L_{i}.xyz")
    R_SC.save_xyz(f"./temp/R_{i}.xyz")
