import numpy as np
import mdtraj as md
import networkx as nx
from itertools import combinations_with_replacement

code_from_letter = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
}


def get_AA(letter, return_graph=False):
    # Returns AA pdb from 1-letter code
    # `letter` is AA 1-letter code
    code = code_from_letter[letter.upper()]
    fname = f"residues/{code}.pdb"
    if return_graph:
        AA = parse_pdb(fname, base_fname="residues/base_residue.pdb", residue=True)
    else:
        AA = md.load(fname)
    return AA


def get_template(core, wing_size, return_graph=False):
    # Returns template pdb from core and wing_size
    # `core` is core abbreviation
    # `wing_size` is the wing size
    fname = f"templates/X{wing_size}-{core}-X{wing_size}.pdb"
    if return_graph:
        template = parse_pdb(fname)
    else:
        template = md.load(fname)
    return template


def get_pdb_compare_dict(pdb_fname):
    """Parses PDB file"""
    compare = dict()
    idx_counter = 0
    with open(pdb_fname) as f:
        for line in f:
            split_line = line.split()
            if (split_line[0] == "HETATM") or (split_line[0] == "ATOM"):
                if "C:" in split_line[-1]:
                    connection = split_line[-1].split("C:")[-1]
                else:
                    connection = None
                compare[idx_counter] = connection
                idx_counter += 1
    return compare


def subgraph_match(G1, G2):
    def element_match(n1, n2):
        if n1["element"] == n2["element"]:
            return True
        return False

    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, element_match)

    if GM.subgraph_is_isomorphic():
        return list(GM.subgraph_isomorphisms_iter())
    else:
        raise ValueError("No matching subgraphs")


def make_bondgraph(top):
    """Returns a bond graph from topology"""
    G = nx.Graph()
    G.add_nodes_from(top.atoms)
    G.add_edges_from(top.bonds)
    element_dict = {atom: atom.element.symbol for atom in G.nodes}
    nx.set_node_attributes(G, element_dict, "element")
    return G


def parse_pdb(pdb_fname, base_fname=None, residue=False):
    """Returns a graph object with available connectivity information"""
    trj = md.load(pdb_fname).center_coordinates()
    G = make_bondgraph(trj.top)
    compare_dict = get_pdb_compare_dict(pdb_fname)

    if len(compare_dict) != trj.top.n_atoms:
        raise ValueError(
            f"Error reading {pdb_fname}. Connect dict ({len(compare_dict)}) != {trj.top.n_atoms} atoms"
        )

    compare_dict = {trj.top.atom(k): v for k, v in compare_dict.items()}
    nx.set_node_attributes(G, compare_dict, "compare")

    # xyz_dict = {atom: xyz for atom, xyz in zip(trj.top.atoms, trj.xyz[0])}
    # nx.set_node_attributes(G, xyz_dict, "xyz")

    if base_fname is not None:
        G_base = parse_pdb(pdb_fname=base_fname, base_fname=None)
        mapping = subgraph_match(G, G_base)[0]
        base_compare = {k: G_base.nodes[v]["compare"] for k, v in mapping.items()}
        nx.set_node_attributes(G, base_compare, "compare")

    if residue:
        compare_temp = nx.get_node_attributes(G, "compare")
        compare = dict()
        for k, v in compare_temp.items():
            if v is None:
                compare[k] = "S"
            else:
                compare[k] = v
        nx.set_node_attributes(G, compare, "compare")

    # edge_idxs = np.empty(shape=(G.number_of_edges(), 2), dtype=np.int)
    # for i, edge in enumerate(G.edges):
    #    edge_idxs[i] = edge[0].index, edge[1].index
    # edge_distances = md.compute_distances(trj, edge_idxs)[0]
    # distance_dict = {edge: dis for edge, dis in zip(G.edges, edge_distances)}
    # nx.set_edge_attributes(G, distance_dict, "distance")
    return G


def rotation_matrix(alpha, beta, gamma):
    """ Returns rotation matrix for transofmorming N x 3 coords (A @ R)"""
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    Rt = R.T
    return Rt


def rotation_matrix_batch(alpha, beta, gamma):
    """ Returns rotation matrix for transofmorming N x 3 coords (A @ R)"""
    n = len(alpha)

    def ef(x):
        a = np.empty(n)
        a.fill(x)
        return a

    Rx = np.array(
        [
            [ef(1), ef(0), ef(0)],
            [ef(0), np.cos(alpha), -np.sin(alpha)],
            [ef(0), np.sin(alpha), np.cos(alpha)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(beta), ef(0), np.sin(beta)],
            [ef(0), ef(1), ef(0)],
            [-np.sin(beta), ef(0), np.cos(beta)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), ef(0)],
            [np.sin(gamma), np.cos(gamma), ef(0)],
            [ef(0), ef(0), ef(1)],
        ]
    )
    # 'li' not 'il' b/c transpose
    Rt = np.einsum("ij...,jk...,kl...->li...", Rz, Ry, Rx)
    return Rt


def get_template_position(template_trj, template_G, name):
    compare_dict = nx.get_node_attributes(template_G, "compare")
    atom_idx = [a.index for a, v in compare_dict.items() if v == name]

    if len(atom_idx) != 1:
        raise ValueError(f"Number of selected atoms {len(atom_idx)} (should be 1)")

    atom_idx = atom_idx[0]
    pos = template_trj.xyz[0, atom_idx]
    return pos


def get_SC(AA_trj, AA_G):
    """Isolate sidechain trj from AA_trj, given the graph (AA_G) """
    compare_dict = nx.get_node_attributes(AA_G, "compare")
    SC_idx = [k.index for k, v in compare_dict.items() if v == "S"]
    SC = AA_trj.atom_slice(SC_idx)
    return SC


def shift_SC(SC_trj, SC_G, rotation, anchor):
    pass


def remove_anchors(template_trj, template_G):
    pass


def get_SC_G(SC_trj, AA_trj, AA_G):
    compare_dict = nx.get_node_attributes(AA_G, "compare")
    for (a0, a1) in AA_G.edges:
        if compare_dict[a0] == "A":
            if compare_dict[a1] == "S":
                anchor = a1
                break
        if compare_dict[a1] == "A":
            if compare_dict[a0] == "S":
                anchor = a0
                break
    SC_G = make_bondgraph(SC_trj.top)
    anchor_dict = dict()
    for atom in SC_trj.top.atoms:
        if atom.name == anchor.name:
            anchor_dict[atom] = True
        else:
            anchor_dict[atom] = False

    nx.set_node_attributes(SC_G, anchor_dict, "anchor")
    return SC_G


def get_SC_rotation(AA_trj, AA_G, target_ev, n=50):
    """ AA_trj is the amino acid trj; AA_G is the amino acid Graph, and target_ev is the target direction"""
    SC = get_SC(AA_trj, AA_G)
    SC.center_coordinates()
    B = SC.xyz[0]

    # Get rotation matricies
    theta = np.linspace(0, np.pi, 50)
    theta = np.array(list(combinations_with_replacement(theta, 3)))
    Rs = rotation_matrix_batch(theta[:, 0], theta[:, 1], theta[:, 2])

    # Generate rotation
    Bs = np.einsum("ij,jk...->ik...", B, Rs)
    SC.xyz = Bs.transpose(2, 0, 1)

    # Compute gyration tensor
    T = md.compute_gyration_tensor(SC)

    # Spectral decomposition
    ls, evs = np.linalg.eig(T)
    largest_ev = np.array([ev[:, i] for ev, i in zip(evs, np.argmax(ls, axis=1))])
    best_idx = np.linalg.norm((largest_ev - target_ev), axis=1).argmin()

    # Find the best rotation matrix
    best_R = Rs[:, :, best_idx]
    return best_R


def get_SC_direction(template_trj, template_G, name):
    """template_trj is template trj; template_G is template graph; and name is the name of the AA (e.g. L1)"""
    compare_dict = nx.get_node_attributes(template_G, "compare")
    C_idx = [k.index for k, v in compare_dict.items() if v == f"C{name}"]
    H_idx = [k.index for k, v in compare_dict.items() if v == f"H{name}"]

    if (len(C_idx) != 1) or (len(H_idx) != 1):
        raise ValueError(f"Cannot find atoms for name {name}")

    C_idx = C_idx[0]
    H_idx = H_idx[0]

    C_xyz = template_trj.xyz[0, C_idx]
    H_xyz = template_trj.xyz[0, H_idx]

    d = H_xyz - C_xyz
    d_norm = d / np.linalg.norm(d)
    return d_norm
