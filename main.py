from genetic_algorithm import GeneticAlgorithm
import click
import numpy as np
import copy

import graph_utils


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
            # print("Found G0, taking the following actions ------------------")
            # print("On point: ", points[-1])
            # Add the intermediate lines to the last point
            points[-1]["following_lines"] = intermediate_lines

            # print("Adding following lines: ", points[-1]["following_lines"])
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

            # print("Intermediate points: ", points[-1]["intermediate_points"])

            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": None}
            points.append(point)

            # print("Started G0, creating point" + str(point))
            # print("---------------------------------------------------------")

            intermediate_lines = [line]

            # breakpoint()

        else:
            # print("Adding to intermediate lines " + str(line))
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


def get_backtrack_from_jump(g_code_output, start_index):
    backtrack_commands = []

    # Step backwards in last point, from start_index to 0
    for i in range(start_index, -1, -1):
        backtrack_commands.append(g_code_output[-1]["path"][i])

    return backtrack_commands


def get_backtrack_commands(g_code_output, closest_group_index, group_path_index):
    backtrack_commands = []

    desired_point = g_code_output[closest_group_index]["path"][group_path_index]
    desired_coord = np.array(get_coords(desired_point))

    # Step backwards in point groups
    # This loop does NOT include the closest group
    for i in range(len(g_code_output) - 1, closest_group_index, -1):
        # Go back to the start of the point
        point = g_code_output[i]
        # Add all of the backtracks in reverse order
        TOLERANCE = 0.2
        for j in range(len(point["backtrack_to_jump"]) - 1, -1, -1):
            backtrack_commands.append(point["backtrack_to_jump"][j])
            # Get the distance between the last backtrack command and the desired point
            command_coord = get_coords(backtrack_commands[-1])
            if command_coord is None:
                continue
            command_coord = np.array(command_coord)

            # if np.linalg.norm(command_coord - desired_coord) < TOLERANCE:
            #     print("Skipping some backtracks")
            #     return backtrack_commands
        for j in range(len(point["path"]) - 1, -1, -1):
            backtrack_commands.append(point["path"][j])
            command_coord = get_coords(backtrack_commands[-1])
            if command_coord is None:
                continue
            command_coord = np.array(command_coord)

            # try:
            #     # if np.linalg.norm(command_coord - desired_coord) < TOLERANCE:
            #     #     print("Skipping some backtracks")
            #     #     return backtrack_commands
            # except:
            #     breakpoint()
        for j in range(len(point["backtrack_from_jump"]) - 1, -1, -1):
            backtrack_commands.append(point["backtrack_from_jump"][j])
            command_coord = get_coords(backtrack_commands[-1])
            if command_coord is None:
                continue
            command_coord = np.array(command_coord)

            # if np.linalg.norm(command_coord - desired_coord) < TOLERANCE:
            #     print("Skipping some backtracks")
            #     return backtrack_commands

    # Step backwards in the closest group to the nearest point
    for i in range(
        len(g_code_output[closest_group_index]["backtrack_to_jump"]) - 1,
        -1,
        -1,
    ):
        backtrack_commands.append(
            g_code_output[closest_group_index]["backtrack_to_jump"][i]
        )
        # if backtrack_commands[-1] == desired_point:
        #     print("Skipping some backtracks")
        #     return backtrack_commands
    for i in range(
        len(g_code_output[closest_group_index]["path"]) - 1, group_path_index - 1, -1
    ):
        backtrack_commands.append(g_code_output[closest_group_index]["path"][i])
        # if backtrack_commands[-1] == desired_point:
        #     print("Skipping some backtracks")
        #     return backtrack_commands

    return backtrack_commands


def get_end_index():
    desired_point = g_code_output[closest_group_index]["path"][group_path_index]
    desired_coord = np.array(get_coords(desired_point))


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

    # Run the genetic algorithm
    while ga.unchanged_gens < 500:
        ga.run_generation()

    print("Converged")
    # Write the output gcode
    best = ga.best

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

    import networkx as nx

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

            # print(f"Block {block_index}")
            # print(f"Closest group index: {closest_group_index}")

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

            # Now, write the output
        with open(output_file, "w") as file:
            for line in pre_g0:
                upper_line = line.upper()
                file.write(upper_line + "\n")

            for sub_block in gcode_output_with_backtrack:

                command = "G0"
                for point in sub_block["final_path"]:
                    file.write(f"{command} X{point[0]} Y{point[1]} F12000\n")

                    command = "G1"

    # for i, path_index in enumerate(best):

    #     # Create a dictionary that stores the different segments of G-Code
    #     g_code_output.append(
    #         {
    #             "backtrack_from_jump": [],
    #             "path": points[path_index]["following_lines"],
    #             "backtrack_to_jump": [],
    #         }
    #     )

    #     if i < len(best) - 1:
    #         # Get the shortest distance between the current point and the next point
    #         closest_group_index, group_1_index, back_track_index = (
    #             ga.get_shortest_distance_with_backtrack(best, i, i + 1)
    #         )

    #         g_code_output[-1]["backtrack_to_jump"] = get_backtrack_commands(
    #             g_code_output, closest_group_index, group_1_index
    #         )

    # Next, remove unnecessary loops
    # print("Removing unnecessary loops")
    # for block_index in range(len(g_code_output)):
    #     # Check for a loop in the path by seeing if the coordinates of the first point are repeated in the path
    #     first_point = get_coords(g_code_output[block_index]["path"][0])

    #     loop_close_index = -1
    #     # Check if the first point is repeated in the path

    #     print("First point", first_point)

    #     for i, line in enumerate(g_code_output[block_index]["path"][1:]):
    #         print("Checking line", line)
    #         if get_coords(line) is not None and np.array_equal(
    #             get_coords(line), first_point
    #         ):
    #             # i+1 is the index of the closing point
    #             loop_close_index = i + 1
    #             break

    #     # If there's not an exact match, check if the last point is within a tolerance of the first point
    #     if loop_close_index == -1:
    #         # Check if the last point is within a tolerance of the first point
    #         last_point = get_coords(g_code_output[block_index]["path"][-1])
    #         if last_point is None:
    #             continue
    #         if np.linalg.norm(last_point - first_point) < 0.1:
    #             print("last point within tolerance")

    #             # Add the last point to the path
    #             g_code_output[block_index]["path"].append(
    #                 g_code_output[block_index]["path"][0]
    #             )
    #             loop_close_index = len(g_code_output[block_index]["path"]) - 1

    #     # If a loop is found, shorten it
    #     if loop_close_index != -1:
    #         breakpoint()
    #         # Construct an array of loop point coords
    #         loop_points = []
    #         for line in g_code_output[block_index]["path"][0 : loop_close_index + 1]:
    #             loop_points.append(get_coords(line))
    #         loop_points = np.array(loop_points)

    #         # Start from the backtrack from jump and find the first point that is in the loop
    #         loop_entry_index = -1
    #         for i, line in enumerate(g_code_output[block_index]["backtrack_from_jump"]):
    #             if (
    #                 get_coords(line) is not None
    #                 and np.where(loop_points == get_coords(line)[None, ...])[0].size > 0
    #             ):
    #                 loop_entry_index = i
    #                 break

    #         assert loop_entry_index != -1, "Loop entry not found"

    #         print("Loop entry found at index", loop_entry_index)

    #         # Start at the end of backtrack to jump and find the first point that is in the loop
    #         loop_exit_index = -1
    #         for i, line in enumerate(g_code_output[block_index]["backtrack_to_jump"]):
    #             if (
    #                 get_coords(line) is not None
    #                 and np.where(loop_points == get_coords(line)[None, ...])[0].size > 0
    #             ):
    #                 loop_exit_index = i
    #                 break

    #         assert loop_exit_index != -1, "Loop exit not found"

    #         print("Loop exit found at index", loop_exit_index)

    #         # First, delete everything from the backtrack from jump to the loop entry
    #         g_code_output[block_index]["backtrack_from_jump"] = g_code_output[
    #             block_index
    #         ]["backtrack_from_jump"][loop_entry_index + 1 :]

    #         # Roll the path so that the loop entry is the first point
    #         g_code_output[block_index]["path"] = np.roll(
    #             g_code_output[block_index]["path"], -loop_close_index
    #         ).tolist()

    #         print("Rolled path by ", -loop_close_index)

    # print("Writing output to", output_file)
    # with open(output_file, "w") as file:
    #     for line in pre_g0:
    #         upper_line = line.upper()
    #         file.write(upper_line + "\n")

    #     for point in g_code_output:
    #         # Write all commands
    #         # Write first element in path - jump, first

    #         for key in ["backtrack_from_jump", "path", "backtrack_to_jump"]:

    #             file.write("; " + key + "\n")
    #             for line in point[key]:
    #                 # Write comment with key
    #                 upper_line = line.upper()
    #                 file.write(upper_line + "\n")

    # print("Output written to", output_file)


if __name__ == "__main__":
    main()
