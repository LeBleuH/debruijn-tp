#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import random
import statistics
import textwrap
import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    has_path
)

# from operator import itemgetter

random.seed(9001)
# from random import randint

from typing import Iterator, Dict, List, Tuple

matplotlib.use("Agg")

__author__ = "Mingjie HUANG"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Mingjie HUANG"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Mingjie HUANG"
__email__ = "mingjie.huang@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r", encoding="utf-8") as f:
        for _, sequence, _, _ in zip(f, f, f, f):
            yield sequence.strip()


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i : i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        graph.add_edge(prefix, suffix, weight=weight)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        for i in range(len(path) - 1):
            if graph.has_edge(path[i], path[i + 1]):
                graph.remove_edge(path[i], path[i + 1])
        nodes_to_remove = []
        if delete_entry_node:
            nodes_to_remove.append(path[0])
        if delete_sink_node:
            nodes_to_remove.append(path[-1])
        for node in path[1:-1]:
            nodes_to_remove.append(node)
        for node in set(nodes_to_remove):
            if graph.has_node(node):
                graph.remove_node(node)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if len(path_list) <= 1:
        return graph

    max_weight = max(weight_avg_list)
    weight_candidates = [
        i for i, w in enumerate(weight_avg_list) if w == max_weight
    ]

    if len(weight_candidates) == 1:
        best_idx = weight_candidates[0]
    else:
        subset_lengths = [path_length[i] for i in weight_candidates]
        max_length = max(subset_lengths)
        length_candidates = [
            weight_candidates[j]
            for j, l in enumerate(subset_lengths)
            if l == max_length
        ]
        if len(length_candidates) == 1:
            best_idx = length_candidates[0]
        else:
            best_idx = random.choice(length_candidates)

    paths_to_remove = [
        path for i, path in enumerate(path_list) if i != best_idx
    ]
    return remove_paths(
        graph, paths_to_remove, delete_entry_node, delete_sink_node
    )


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    if len(paths) <= 1:
        return graph
    lengths = [len(p) - 1 for p in paths]
    weights = [path_average_weight(graph, p) for p in paths]
    return select_best_path(
        graph, paths, lengths, weights,
        delete_entry_node=False, delete_sink_node=False
    )


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        found = False
        nodes = list(graph.nodes())
        for a in nodes:
            if a not in graph:
                continue
            if graph.out_degree(a) <= 1:
                continue
            for b in nodes:
                if a == b or b not in graph:
                    continue
                if not has_path(graph, a, b):
                    continue
                paths = list(all_simple_paths(graph, a, b))
                if len(paths) >= 2:
                    graph = solve_bubble(graph, a, b)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        found = False
        current_starts = get_starting_nodes(graph)
        for n in list(graph.nodes()):
            if graph.in_degree(n) <= 1:
                continue
            paths_by_start = {}
            for s in current_starts:
                if has_path(graph, s, n):
                    for p in all_simple_paths(graph, s, n):
                        paths_by_start[s] = p
                        break
            if len(paths_by_start) >= 2:
                paths = list(paths_by_start.values())
                lengths = [len(p) - 1 for p in paths]
                weights = [path_average_weight(graph, p) for p in paths]
                graph = select_best_path(
                    graph, paths, lengths, weights,
                    delete_entry_node=True, delete_sink_node=False
                )
                found = True
                break
        if not found:
            break
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        found = False
        current_ends = get_sink_nodes(graph)
        for n in list(graph.nodes()):
            if graph.out_degree(n) <= 1:
                continue
            paths_by_end = {}
            for e in current_ends:
                if has_path(graph, n, e):
                    for p in all_simple_paths(graph, n, e):
                        paths_by_end[e] = p
                        break
            if len(paths_by_end) >= 2:
                paths = list(paths_by_end.values())
                lengths = [len(p) - 1 for p in paths]
                weights = [path_average_weight(graph, p) for p in paths]
                graph = select_best_path(
                    graph, paths, lengths, weights,
                    delete_entry_node=False, delete_sink_node=True
                )
                found = True
                break
        if not found:
            break
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes() if graph.in_degree(node) == 0]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List[Tuple[str, int]]:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for sink in ending_nodes:
            if has_path(graph, start, sink):
                for path in all_simple_paths(graph, start, sink):
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]
                    contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contigs_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (contig, length) in enumerate(contigs_list):
            f.write(f">contig_{i} len={length}\n")
            f.write(textwrap.fill(contig, width=80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    # fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)

    graph = build_graph(kmer_dict)

    graph = simplify_bubbles(graph)

    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)

    ending_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, ending_nodes)

    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, ending_nodes)

    save_contigs(contigs, args.output_file)

    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
