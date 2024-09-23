import numpy as np
import click
from pathlib import Path
import struct
import pickle
from dataclasses import dataclass
import graph_utils
import networkx as nx
import subprocess

from typing import List, Optional


@dataclass
class GCodePoint:
    x: float
    y: float
    z: Optional[float] = None
    f: Optional[float] = None

    def __init__(self, x, y, z=None, f=None):
        self.x = x
        self.y = y
        self.z = z
        self.f = f


@dataclass
class GCodeBlock:
    points: List[GCodePoint]

    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"GCodeBlock(points={self.points})"


def get_coords(point):
    x = point["x"]
    y = point["y"]

    return (x, y)


def decode_solution(data):
    """Output a list"""
    data = data["solution"]

    output = []
    for block in data:
        block = block["points"]
        points = [None] * len(block)
        for i, point in enumerate(block):
            points[i] = get_coords(point)

        output.append(
            {
                "path": points,
            }
        )

    return output


def optimize_gcode(input_file: str, output_file: str, origin: tuple[float, float]):

    PATH_TO_RUST_OPTIMIZER = "gcode-optimizer"

    # Get just the directory of the output file
    output_dir = Path(output_file).parent

    # Run the rust optimizer
    command = f"{PATH_TO_RUST_OPTIMIZER} --input {input_file} --output {output_dir}"

    # Get the feedrate from the input file
    # Right now, this just reads the first line with an F in it
    # Okay for etch a sketch code, but not for general gcode
    # Set a default
    feedrate = 12000
    with open(input_file, "r") as file:
        for line in file:
            if "F" in line:
                feedrate = float(line.split("F")[1].strip())
                break

    # Run the command
    proc = subprocess.Popen(command, shell=True)
    proc.wait()

    input_name = Path(input_file).stem
    solution_file = output_dir / f"{input_name}_solution.bin"
    backtrack_file = output_dir / f"{input_name}_backtrack.bin"

    # Open solution.bin
    with open(solution_file, "rb") as file:
        received_points = pickle.load(file)

    # Open backtrack.bin
    with open(backtrack_file, "rb") as file:
        backtrack_points = pickle.load(file)

    g_code_output = decode_solution(received_points)

    for i, block in enumerate(g_code_output):

        graph = graph_utils.graph_from_path(block["path"])

        g_code_output[i]["graph"] = graph

    # Convert pickled data to list of GCodeBlocks
    combined_graph = nx.Graph()
    # Add origin as the first point
    combined_graph.add_node(origin)
    next_group_index = 0
    gcode_output_with_backtrack = []
    last_node = origin
    for block_index, block in enumerate(g_code_output):
        combined_graph = nx.compose(combined_graph, block["graph"])

        new_edge = (
            block["path"][next_group_index],
            last_node,
        )
        combined_graph.add_edge(*new_edge)

        if block_index < len(g_code_output) - 1:
            # Fully explore this block
            block["use_shortest_path"] = False
            block["start"] = block["path"][next_group_index]

            closest_group_index, group_1_index, next_group_index = backtrack_points[
                block_index
            ]

            # Next jump is in this group
            if closest_group_index == block_index:
                # There is no backtrack
                block["end"] = block["path"][group_1_index]

                gcode_output_with_backtrack.append(block)

            else:

                # Need to end up at the previous group
                block["end"] = block["start"]

                gcode_output_with_backtrack.append(block)

                # Find the shortest path to the next group
                point = g_code_output[closest_group_index]["path"][group_1_index]

                shortest_path = graph_utils.get_shortest_path(
                    combined_graph, block["end"], point
                )

                # Add the backtrack commands
                backtrack_block = {
                    "final_path": shortest_path,
                    "end": point,
                    "start": block["end"],
                }

                gcode_output_with_backtrack.append(backtrack_block)

            last_node = gcode_output_with_backtrack[-1]["end"]

        else:
            # Last block
            block["use_shortest_path"] = False
            block["start"] = block["path"][next_group_index]
            block["end"] = block["path"][-1]

            gcode_output_with_backtrack.append(block)

            # Get shortest path to origin
            shortest_path = graph_utils.get_shortest_path(
                combined_graph, block["end"], origin
            )

            # Add the backtrack commands
            backtrack_block = {
                "final_path": shortest_path,
                "end": origin,
                "start": block["end"],
            }

            gcode_output_with_backtrack.append(backtrack_block)

        # Generate the path now
        for i, block_in in enumerate(gcode_output_with_backtrack):

            if i == 0:
                print("Block in ", block_in["start"], block_in["end"])

            if not "final_path" in block_in:
                if block_in["use_shortest_path"]:
                    path = graph_utils.get_shortest_path(
                        block_in["graph"],
                        block_in["start"],
                        block_in["end"],
                    )
                else:
                    path = graph_utils.get_traversing_path(
                        block_in["graph"],
                        block_in["start"],
                        block_in["end"],
                    )

                block_in["final_path"] = path

                gcode_output_with_backtrack[i] = block_in

        # Write the output

        buffer = []

        with open(output_file, "w") as file:
            for sub_block in gcode_output_with_backtrack:
                command = "G0"
                for point in sub_block["final_path"]:
                    buffer.append(f"{command} X{point[0]} Y{point[1]} F{feedrate}\n")
                    command = "G1"
            file.writelines(buffer)

@click.command()
@click.option(
    "--input_file", prompt="Path to input binary file", help="Path to input binary file"
)
@click.option(
    "--output_file",
    prompt="Path to output gcode file",
    help="Path to output gcode file",
)
# Origin arguments
@click.option(
    "--origin_x",
    prompt="X coordinate of origin",
    help="X coordinate of origin",
    default=0.0,
)
@click.option(
    "--origin_y",
    prompt="Y coordinate of origin",
    help="Y coordinate of origin",
    default=0.0,
)
def main(input_file: str, output_file: str, origin_x: float, origin_y: float):
    optimize_gcode(input_file, output_file, (origin_x, origin_y))


if __name__ == "__main__":
    main()
