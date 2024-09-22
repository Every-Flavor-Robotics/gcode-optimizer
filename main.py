from genetic_algorithm import GeneticAlgorithm
import click
import numpy as np
import copy

import graph_utils
import networkx as nx
import time


def get_coords(line: str) -> np.ndarray:
    x = 0
    y = 0

    # Check if line starts with "G"
    if not line.startswith("g"):
        return None

    if "x" in line:
        x = float(line.split("x")[1].split(" ")[0])
    if "y" in line:
        y = float(line.split("y")[1].split(" ")[0])

    return np.array([x, y])


def read_gcode(input_path):
    with open(input_path, "r") as file:
        lines = file.readlines()

    points = []

    pre_g0 = []

    # First loop to fill up pre_g0
    while lines:
        line = lines.pop(0).lower().strip()
        print(line)

        if line.startswith("g0"):
            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": None}
            # print("Started G0, creating point" + str(point))
            points.append(point)
            break
        else:
            # print("Adding to pre_g0 " + str(line))
            pre_g0.append(line)

        # breakpoint()

    intermediate_lines = [line]
    while lines:
        line = lines.pop(0).lower().strip()
        # print("Next line: ", line)

        if line.startswith("g0"):
            # Add the intermediate lines to the last point
            points[-1]["following_lines"] = intermediate_lines

            # Get end coordinates of the last point
            points[-1]["intermediate_points"] = []
            for intermediate_line in intermediate_lines:
                coords = get_coords(intermediate_line)
                if coords is not None:
                    points[-1]["intermediate_points"].append(
                        get_coords(intermediate_line)
                    )
            points[-1]["intermediate_points"] = np.array(
                points[-1]["intermediate_points"]
            )

            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": None}
            points.append(point)

            intermediate_lines = [line]

            # breakpoint()

        else:
            intermediate_lines.append(line)

    # Add the intermediate lines to the last point
    points[-1]["following_lines"] = intermediate_lines
    # Get end coordinates of the last point
    points[-1]["intermediate_points"] = []
    for intermediate_line in intermediate_lines:
        coords = get_coords(intermediate_line)
        if coords is not None:
            points[-1]["intermediate_points"].append(get_coords(intermediate_line))
    points[-1]["intermediate_points"] = np.array(points[-1]["intermediate_points"])

    return pre_g0, points


@click.command()
@click.option(
    "--input_file", prompt="Path to input gcode file", help="Path to input gcode file"
)
@click.option(
    "--output_file",
    prompt="Path to output gcode file",
    help="Path to output gcode file",
)
def main(input_file, output_file):
    # Create an instance of the GeneticAlgorithm class
    ga = GeneticAlgorithm()

    # Open the input file and read the gcode
    pre_g0, points = read_gcode(input_file)
    ga.init(points)

    total_time = 0
    loops = 0

    # Run the genetic algorithm
    while ga.unchanged_gens < 500:
        start = time.time()
        ga.run_generation()
        end = time.time()

        # Total time in millis
        total_time += (end - start) * 1000
        loops += 1

    print("Total time: ", total_time)

    print("Converged")
    # Write the output gcode
    best = ga.best
    print(best)
    breakpoint()

    print("Generating G-Code with backtracking...")
    g_code_output = []
    back_track_index = 0

    for i, path_index in enumerate(best):
        path = points[path_index]["intermediate_points"]
        # Convert path into a list of tuples
        path = [tuple(point) for point in path]
        graph = graph_utils.graph_from_path(path)

        g_code_output.append(
            {
                "path": path,
                "graph": graph,
            }
        )

    combined_graph = nx.Graph()
    # Add (0,0) as the first point
    combined_graph.add_node((0, 0))
    next_group_index = 0
    gcode_output_with_backtrack = []
    last_node = (0, 0)
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

            closest_group_index, group_1_index, next_group_index = (
                ga.get_shortest_distance_with_backtrack(
                    best, block_index, block_index + 1
                )
            )
            breakpoint()

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

            # Get shortest path to (0,0)
            shortest_path = graph_utils.get_shortest_path(
                combined_graph, block["end"], (0, 0)
            )

            # Add the backtrack commands
            backtrack_block = {
                "final_path": shortest_path,
                "end": (0, 0),
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
        with open(output_file, "w") as file:
            for line in pre_g0:
                upper_line = line.upper()
                file.write(upper_line + "\n")

            for sub_block in gcode_output_with_backtrack:

                command = "G0"
                for point in sub_block["final_path"]:
                    file.write(f"{command} X{point[0]} Y{point[1]} F12000\n")

                    command = "G1"

    # print("Output written to", output_file)


if __name__ == "__main__":
    main()
