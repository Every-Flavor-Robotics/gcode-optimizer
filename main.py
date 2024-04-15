from genetic_algorithm import GeneticAlgorithm
import click
import numpy as np


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
            print("Started G0, creating point" + str(point))
            points.append(point)
            break
        else:
            print("Adding to pre_g0 " + str(line))
            pre_g0.append(line)

        # breakpoint()

    intermediate_lines = [line]
    while lines:
        line = lines.pop(0).lower().strip()
        print("Next line: ", line)

        if line.startswith("g0"):
            print("Found G0, taking the following actions ------------------")
            print("On point: ", points[-1])
            # Add the intermediate lines to the last point
            points[-1]["following_lines"] = intermediate_lines

            print("Adding following lines: ", points[-1]["following_lines"])
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

            print("Intermediate points: ", points[-1]["intermediate_points"])

            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": None}
            points.append(point)

            print("Started G0, creating point" + str(point))
            print("---------------------------------------------------------")

            intermediate_lines = [line]

            # breakpoint()

        else:
            print("Adding to intermediate lines " + str(line))
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
    "--output", prompt="Path to output gcode file", help="Path to output gcode file"
)
def main(input_file, output):
    # Create an instance of the GeneticAlgorithm class
    ga = GeneticAlgorithm()

    # Open the input file and read the gcode

    pre_g0, points = read_gcode(input_file)
    ga.init(points)

    # Run the genetic algorithm
    while ga.unchanged_gens < 50:
        ga.run_generation()

    print("Converged")
    # Write the output gcode
    best = ga.best

    with open(output, "w") as file:
        for line in pre_g0:
            upper_line = line.upper()
            file.write(upper_line + "\n")

        breakpoint()
        backtrack_commands = []
        for i, index in enumerate(best):
            point = points[index]

            # Write the G0 command
            line = point["following_lines"][0]

            upper_line = line.upper()
            file.write("; Jump\n" + upper_line + "\n")

            # Write backtracking commands
            for backtrack_command in backtrack_commands:
                backtrack_command = backtrack_command.upper()
                file.write(backtrack_command + "\n")

            # Add command
            file.write("; Done backtracking after jump\n")
            # Write remaining intermediate
            for line in point["following_lines"][1:]:
                upper_line = line.upper()
                file.write(upper_line + "\n")

            # Backtrack to the nearest point to the next point
            # No need to backtrack for the last point
            if i < len(best) - 1:
                file.write("; Backtracking to next jump\n")

                next_point = points[best[i + 1]]
                _, group_1_index, group_2_index = ga.get_shortest_distance(
                    point["intermediate_points"], next_point["intermediate_points"]
                )

                # Step backwards to the nearest point in group 1 and write the gcode
                for j in range(
                    len(point["following_lines"]) - 1, group_1_index - 1, -1
                ):
                    upper_line = point["following_lines"][j].upper()
                    file.write(upper_line + "\n")

            backtrack_commands = []
            # Step backwards to the nearest point in group 2 and write the gcode
            for j in range(group_2_index, -1, -1):
                backtrack_commands.append(next_point["following_lines"][j])

    print("Output written to", output)


if __name__ == "__main__":
    main()
