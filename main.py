from GeneticAlgorithm import GeneticAlgorithm
import click
import numpy as np


def get_coords(line: str) -> np.ndarray:
    x = 0
    y = 0

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

        if line.startswith("g0"):
            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": [line]}
            points.append(point)
            break
        else:
            pre_g0.append(line)

    intermediate_lines = []
    while lines:
        line = lines.pop(0).lower().strip()

        if line.startswith("g0"):
            # Add the intermediate lines to the last point
            points[-1]["following_lines"].extend(intermediate_lines)
            # Get end coordinates of the last point
            end_coord = get_coords(points[-1]["following_lines"][-1])
            points[-1]["end_coord"] = end_coord

            # Create a new point
            # Add the coordinates of the line to the point
            # Add the line to the following_lines of the point
            point = {"start_coord": get_coords(line), "following_lines": [line]}
            points.append(point)

            intermediate_lines = []
        else:
            intermediate_lines.append(line)

    # Add the intermediate lines to the last point
    points[-1]["following_lines"].extend(intermediate_lines)
    # Get end coordinates of the last point
    end_coord = get_coords(points[-1]["following_lines"][-1])
    points[-1]["end_coord"] = end_coord

    return pre_g0, points


@click.command()
@click.option(
    "--input", prompt="Path to input gcode file", help="Path to input gcode file"
)
@click.option(
    "--output", prompt="Path to output gcode file", help="Path to output gcode file"
)
def main(input, output):
    # Create an instance of the GeneticAlgorithm class
    ga = GeneticAlgorithm()

    # Open the input file and read the gcode

    pre_g0, points = read_gcode(input)
    ga.init(points)

    # Run the genetic algorithm
    while ga.unchanged_gens < 10:
        ga.run_generation()

    print("Converged")
    # Write the output gcode
    best = ga.best

    with open(output, "w") as file:
        for line in pre_g0:
            upper_line = line.upper()
            file.write(upper_line + "\n")

        for index in best:
            point = points[index]
            for line in point["following_lines"]:
                upper_line = line.upper()
                file.write(upper_line + "\n")

    print("Output written to", output)


if __name__ == "__main__":
    main()
