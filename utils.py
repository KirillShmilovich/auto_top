import numpy as np
import mdtraj as md
import networkx as nx

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
