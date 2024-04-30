import networkx as nx

import matplotlib.pyplot as plt


def get_odd_degree_count(graph):
    degrees = graph.degree()

    odd_degree_count = 0
    for node, degree in degrees:
        if degree % 2 == 1:
            odd_degree_count += 1

    return odd_degree_count


def get_degree_one_count(graph):
    degrees = graph.degree()

    degree_one_count = 0
    for node, degree in degrees:
        if degree == 1:
            degree_one_count += 1

    return degree_one_count


def graph_from_path(path):
    graph = nx.DiGraph()

    # Create an edge list - each edge is a pair of coordinates that are adjacent in the path
    edge_list = []
    for i in range(len(path) - 1):
        edge_list.append((path[i], path[i + 1]))

    # Remove self-loops
    edge_list = [edge for edge in edge_list if edge[0] != edge[1]]

    graph.add_nodes_from(path)
    graph.add_edges_from(edge_list)

    # Make graph undirected
    graph = graph.to_undirected()

    return graph


def get_connection_path(node, graph):

    # Plot graph
    # nx.draw(graph, with_labels=True)
    # plt.show()

    # breakpoint()

    assert graph.degree(node) == 1
    # Construct the path from the node to a node with degree >2
    path = [node]

    # Get the current node
    # The node must have degree 1
    neighbors = [neighbor for neighbor in graph.neighbors(node)]
    current_node = neighbors[0]
    while graph.degree(current_node) == 2:
        # Add the current node to the path
        path.append(current_node)

        # Get the neighbor that is not the previous node
        for neighbor in graph.neighbors(current_node):
            if neighbor != path[-2]:
                break

        current_node = neighbor

    # Add the last node to the path
    path.append(current_node)

    path = graph_from_path(path)
    path.graph["entry_point"] = current_node
    return path


def get_eulerian_path(graph):
    # Check if the graph has an Eulerian path
    odd_degree_count = get_odd_degree_count(graph)

    if odd_degree_count == 0:
        return graph

    # If zero, graph is a Eulerian tour
    # If non-zero, we update nodes to make it a Eulerian tour
    # If zero, means we have created Eulerian tour
    while odd_degree_count != 2 and odd_degree_count != 0:
        # Find a node with degree == 1
        for node, degree in graph.degree():
            if degree == 1:
                break

        # Get the connection path
        # This path defines the path from the node with degree 1 to a node with degree >2
        connection_path = get_connection_path(node, graph)

        # We need to remove the nodes in the connection path from the original graph
        # Replace the connection path with a node
        # First, let's get the neighbor list of "entry_point"
        # Remove the neighbor from connection path
        neighbors = [
            neighbor
            for neighbor in graph.neighbors(
                connection_path.graph.get("entry_point", None)
            )
        ]

        for i, neighbor in enumerate(neighbors):
            if neighbor in connection_path.nodes:
                neighbors.pop(i)
                break

        # Next, delete all the nodes in the connection path from the original graph
        graph.remove_nodes_from(connection_path.nodes)

        # Add the new node
        graph.add_node(connection_path)
        # Create an edge list between the new node and the neighbors
        edge_list = []
        for neighbor in neighbors:
            edge_list.append((connection_path, neighbor))

        graph.add_edges_from(edge_list)

        # Get the new odd degree count
        # Plot the graph
        # nx.draw(graph, with_labels=True)
        # plt.show()

        odd_degree_count = get_odd_degree_count(graph)

    return graph


def graph_contains(graph, node):
    for n in graph.nodes:
        if n == node:
            return True
        elif isinstance(n, nx.Graph):
            if graph_contains(n, node):
                return True
    return False


def graph_where(graph, node):
    for i, n in enumerate(graph.nodes):
        if n == node:
            return n
        elif isinstance(n, nx.Graph):
            if graph_contains(n, node):
                return n
    return None


def get_coords(node):
    # If the node is a tuple, return the tuple
    if isinstance(node, tuple):
        return node

    # If the node is a graph, return the entry point
    if isinstance(node, nx.Graph):
        # Get the entry point
        return get_coords(node.graph.get("entry_point", None))


def distance(node1, node2, attributes):
    node_1_pos = get_coords(node1)
    node_2_pos = get_coords(node2)

    return abs(node_1_pos[0] - node_2_pos[0]) + abs(node_1_pos[1] - node_2_pos[1])


def flatten_path(path):
    for node in path["main"]:
        # If the node is a graph, recursively flatten
        if isinstance(node, nx.Graph):
            flat_path.extend(flatten_path(path[node]))
        else:
            flat_path.append(node)
    return flat_path


def add_subpath_to_path(path, node, sub_path):
    if node not in path:
        path[node] = sub_path

    # Do nothing otherwise

    return path


def get_traversing_path(graph, start_node, end_node):

    # Create a new graph if path is None
    path = {"main": []}

    # Get keys for the start and end nodes
    # These could point to a node or a graph
    start_node_key = graph_where(graph, start_node)
    end_node_key = graph_where(graph, end_node)

    used_edges

    # Check if start_node key is a graph
    if isinstance(start_node_key, nx.Graph):
        # Find the path from the start node to the entry point
        start_path = get_traversing_path(
            start_node_key, start_node, start_node_key.graph["entry_point"]
        )

        path = add_subpath_to_path(path, start_node_key, start_path)
    else:
        path["main"].append(start_node_key)

    # Plot the graph
    # nx.draw(graph, with_labels=True)
    # plt.show()
    # Check if the start node has multiple neighbors
    neighbors = [neighbor for neighbor in graph.neighbors(start_node_key)]
    if len(neighbors) == 0:
        return path  # No neighbors, return the path
    if len(neighbors) > 1:
        # Get shortest path from start node to end node
        sub_path = nx.shortest_path(graph, start_node_key, end_node_key)[0]

        # Select neighbor in not in sub_path
        for neighbor in neighbors:
            if neighbor not in sub_path:
                break

    else:
        neighbor = neighbors[0]
    current_node = neighbor
    while current_node != end_node_key:
        # If its a graph
        if isinstance(current_node, nx.Graph):

            # If it's not in main, we need to recursively traverse
            # Also, do traverse the end node early
            if not current_node in path["main"] and current_node != end_node_key:
                sub_path = get_traversing_path(
                    current_node,
                    current_node.graph["entry_point"],
                    current_node.graph["entry_point"],
                )
                path = add_subpath_to_path(path, current_node, sub_path)
            # If it is in main, simply add the entry point and continue
            else:
                path["main"].append(current_node.graph["entry_point"])
        else:
            path["main"].append(current_node)

        # Get the neighbors of the start node key
        neighbors = [neighbor for neighbor in graph.neighbors(current_node)]

        # We've reached the end of a path, start walking backwards
        if len(neighbors) == 1:
            current_node = neighbors[0]
            continue

        # Get the neighbor that is not the previous node
        for neighbor in neighbors:
            if neighbor != path["main"][-2]:
                break

        current_node = neighbor

    # First, check if end_node_key has additional nodes beyond
    end_node_degree = graph.degree(current_node)
    if end_node_degree > 1:

        # breakpoint()
        # First add the end node to the graph
        # If its a graph, add the entry point
        if isinstance(current_node, nx.Graph):
            path["main"].append(current_node.graph["entry_point"])
        else:
            path["main"].append(current_node)

        # Continue walking past the end node until we get back
        neighbors = [neighbor for neighbor in graph.neighbors(current_node)]
        for neighbor in neighbors:
            if neighbor != path["main"][-2]:
                break

        current_node = neighbor
        while current_node != end_node_key:
            # If its a graph
            if isinstance(current_node, nx.Graph):

                # If it's not in main, we need to recursively traverse
                if not current_node in path["main"]:
                    sub_path = get_traversing_path(
                        current_node,
                        current_node.graph["entry_point"],
                        current_node.graph["entry_point"],
                    )
                    path = add_subpath_to_path(path, current_node, sub_path)
                # If it is in main, simply add the entry point and continue
                else:
                    path["main"].append(current_node.graph["entry_point"])
            else:
                path["main"].append(current_node)

            # Get the neighbors of the start node key
            neighbors = [neighbor for neighbor in graph.neighbors(current_node)]

            # We've reached the end of a path, start walking backwards
            if len(neighbors) == 1:
                current_node = neighbors[0]
                continue

            # Get the neighbor that is not the previous node
            for neighbor in neighbors:
                # If neighbor is a graph instance
                if isinstance(neighbor, nx.Graph):
                    if neighbor.graph["entry_point"] != path["main"][-2]:
                        break
                elif neighbor != path["main"][-2]:
                    break

            current_node = neighbor

    # Next, check if the end_node_key is a graph
    if isinstance(end_node_key, nx.Graph):
        # Very specific edge case
        # If the end node key is the same as the start node key
        # And the end node is the entry point of the end node key
        # Skip the loop
        if (
            start_node_key == end_node_key
            and end_node == end_node_key.graph["entry_point"]
        ):
            # Add the end node to the path
            path["main"].append(end_node_key.graph["entry_point"])
            return path
        end_path = get_traversing_path(
            end_node_key, end_node_key.graph["entry_point"], end_node
        )

        path = add_subpath_to_path(path, end_node_key, end_path)

    else:
        path["main"].append(end_node_key)

    return path


def edge_in_list(edge, edge_list):
    edge_reversed = (edge[1], edge[0])

    return edge in edge_list or edge_reversed in edge_list


def construct_tour(eulerian_graph):
    # View the graph
    # print(eulerian_graph)
    # nx.draw(eulerian_graph, with_labels=True)
    # plt.show()
    # breakpoint()

    if len(eulerian_graph.nodes) == 0:
        return {"tour": [], "is_tour": True}
    if len(eulerian_graph.nodes) == 1:
        return {"tour": list(eulerian_graph.nodes), "is_tour": True}

    tour = []
    # Get the degree of each node
    degrees = eulerian_graph.degree()
    # If any of the degrees are 1, we need to start at that node
    start_node = None

    is_tour = True
    for node, degree in degrees:
        if degree == 1:
            start_node = node
            is_tour = False
            break

    # If no node has degree 1, start at any node
    if start_node is None:
        start_node = list(eulerian_graph.nodes)[0]

    # Create a list of already visited edges
    visited_edges = []
    # First pass of tour
    cur_node = start_node
    while True:
        neighbors = [neighbor for neighbor in eulerian_graph.neighbors(cur_node)]
        selected_neighbor = None
        edge = None
        for neighbor in neighbors:
            edge = (cur_node, neighbor)
            if not edge_in_list(edge, visited_edges):
                selected_neighbor = neighbor
                break

        if selected_neighbor is None:
            break

        visited_edges.append(edge)
        tour.append((cur_node, neighbor))
        cur_node = neighbor

    while len(visited_edges) < len(eulerian_graph.edges):

        # Second pass of tour
        for i, (start_node, end_node) in enumerate(tour):
            # Check if start node has any unvisited edges
            neighbors = [neighbor for neighbor in eulerian_graph.neighbors(start_node)]

            new_edge = None
            # Iterate through all neighbors and check if the edge is visited
            for neighbor in neighbors:
                edge = (start_node, neighbor)
                if not edge_in_list(edge, visited_edges):
                    new_edge = edge
                    break

            # If all edges are visted for this node, move to the next edge
            if new_edge is None:
                continue

            # Insert the new edge into the tour
            sub_tour = [(start_node, neighbor)]
            visited_edges.append(new_edge)
            cur_node = neighbor
            while cur_node != start_node:
                neighbors = [
                    neighbor for neighbor in eulerian_graph.neighbors(cur_node)
                ]
                selected_neighbor = None
                edge = None
                for neighbor in neighbors:
                    edge = (cur_node, neighbor)
                    if not edge_in_list(edge, visited_edges):
                        selected_neighbor = neighbor
                        break

                if selected_neighbor is None:
                    break

                visited_edges.append(edge)
                sub_tour.append((cur_node, neighbor))
                cur_node = neighbor

            # Insert the sub tour into the main tour
            tour = tour[:i] + sub_tour + tour[i:]
            break
            # Add the start node back

    # Next, recursively construct tours of any subgraphs
    expanded_nodes = []
    output = {
        "tour": [],
        "is_tour": is_tour,
    }
    for i, (start_node, end_node) in enumerate(tour):
        if isinstance(start_node, nx.Graph):
            # Expand if not already expanded
            if start_node in expanded_nodes:
                # Add entry point
                output["tour"].append(start_node.graph["entry_point"])
            else:
                # Generate the sub tour
                sub_tour = construct_tour(start_node)
                output["tour"].append(start_node)
                add_subpath_to_path(output, start_node, sub_tour)
                expanded_nodes.append(start_node)
        else:
            output["tour"].append(start_node)

    # Process the last node
    try:
        if isinstance(end_node, nx.Graph):
            # if end_node in expanded_nodes:
            #     # Add entry point
            #     output["tour"].append(end_node.graph["entry_point"])
            # else:
            sub_tour = construct_tour(end_node)
            output["tour"].append(end_node)
            add_subpath_to_path(output, end_node, sub_tour)
            expanded_nodes.append(end_node)
        else:
            output["tour"].append(end_node)
    except:
        breakpoint()

    return output


# def retrieve_path(node_vistation_count, node, tour):

#     # If not a list, there is only one path
#     if not isinstance(tour[node], list):
#         node_vistation_count[node] = 1
#         return tour[node]

#     # If the node has been visited before, increment the count
#     if node in node_vistation_count:
#         node_vistation_count[node] += 1
#     else:
#         node_vistation_count[node] = 1

#     return tour[node][node_vistation_count[node] - 1]


def get_shortest_path(graph, start, end):
    if start == end:
        return [start]

    # Get the shortest path from start to end
    start_node = graph_where(graph, start)
    end_node = graph_where(graph, end)

    # Get the shortest path
    shortest_path = nx.shortest_path(graph, start_node, end_node, weight=distance)

    expanded_shortest_path = []
    for i, node in enumerate(shortest_path):
        if isinstance(node, nx.Graph):
            if i == 0:
                sub_path_start = start
                sub_path_end = node.graph["entry_point"]
            elif i == len(shortest_path) - 1:
                sub_path_start = node.graph["entry_point"]
                sub_path_end = end
            else:
                sub_path_start = node.graph["entry_point"]
                sub_path_end = node.graph["entry_point"]

            sub_path = get_shortest_path(node, start, end)
            expanded_shortest_path.extend(sub_path)
        else:
            expanded_shortest_path.append(node)

    return expanded_shortest_path


def _get_traversing_path_helper(graph, tour, start, end):

    # Find the start and end node in the tour
    # This will either be the node itself or a graph containing the node
    start_node = graph_where(graph, start)
    end_node = graph_where(graph, end)

    # breakpoint()

    # # Create a unique identifier for each subgraph that needs to be traversed
    # id_to_graph = {}
    # node_vistation_count = {}
    # for i, node in enumerate(tour["tour"]):
    #     if isinstance(node, nx.Graph):
    #         id_to_graph[i] = node
    #         tour["tour"][i] = i
    #         tour[i] = tour[node].pop(0)

    # If this is a tour, the whole thing is a loop
    # We need to simply start at the start node and traverse the tour
    if tour["is_tour"]:
        # Confirm that it loops as a tour
        assert tour["tour"][0] == tour["tour"][-1]

        # Find the first instance of the start node
        start_index = tour["tour"].index(start_node)
        traversing_path = tour["tour"][start_index:]
        # Start at 1 to avoid the start node, which is already in the path
        # Include start node in the path
        traversing_path.extend(tour["tour"][1:start_index])

        # Find the shortest path from the start node to the end node
        shortest_path = nx.shortest_path(graph, start_node, end_node, weight=distance)

        # Add the shortest path to the traversing path
        traversing_path.extend(shortest_path)

    else:
        # This is a path, not a tour
        # Split trajectory into the following parts:
        # 1. Start node to the start node key
        # 2. Start node key to end node key
        # 3. End node key to end node

        # Find the start node key
        start_index = tour["tour"].index(start_node)
        end_index = tour["tour"].index(end_node)

        # Flip the path if the start index is greater than the end index
        if start_index > end_index:
            tour["tour"] = tour["tour"][::-1]
            start_index = tour["tour"].index(start_node)
            end_index = tour["tour"].index(end_node)

        start_to_path_begin = tour["tour"][: start_index + 1][::-1]
        start_to_end = tour["tour"][start_index : end_index + 1]
        end_to_path_end = tour["tour"][end_index:]

        if len(start_to_path_begin) == 0:
            path_begin_to_start = []
        else:
            # Get shortest path back
            path_begin_to_start = nx.shortest_path(
                graph, start_to_path_begin[-1], start_node, weight=distance
            )
        if len(end_to_path_end) == 0:
            path_end_to_end = []
        else:
            path_end_to_end = nx.shortest_path(
                graph, end_to_path_end[-1], end_node, weight=distance
            )

        # breakpoint()
        # Go to path begin
        traversing_path = start_to_path_begin
        # Return to start node
        # Start at 1 to avoid the start node, which is already in the path
        traversing_path.extend(path_begin_to_start[1:])
        # Go to end node
        traversing_path.extend(start_to_end[1:])
        # Go to path end
        traversing_path.extend(end_to_path_end[1:])
        # Return to end node
        traversing_path.extend(path_end_to_end[1:])

    # breakpoint()
    # Now, recursively traverse the subgraphs within the path
    new_traversing_path = []
    expanded_nodes = []
    for i, node in enumerate(traversing_path):
        # If it's a graph
        if isinstance(node, nx.Graph):
            if len(traversing_path) == 1:
                # If the graph is the only node in the path
                sub_path_start = start
                sub_path_end = end

                sub_path = get_traversing_path(
                    node, tour[node], sub_path_start, sub_path_end
                )
            else:
                # Define and start of the sub path depending
                # on whether the graph is in the beginning, middle, or end of path
                if i == 0:
                    sub_path_start = start
                    sub_path_end = node.graph["entry_point"]
                elif i == len(traversing_path) - 1:
                    sub_path_start = node.graph["entry_point"]
                    sub_path_end = end
                else:
                    sub_path_start = node.graph["entry_point"]
                    sub_path_end = node.graph["entry_point"]

                if (node in expanded_nodes or node == end_node) and i != len(
                    traversing_path
                ) - 1:
                    sub_path = get_shortest_path(node, sub_path_start, sub_path_end)
                else:
                    sub_path = get_traversing_path(
                        node, tour[node], sub_path_start, sub_path_end
                    )
                    expanded_nodes.append(node)

            new_traversing_path.extend(sub_path)
        else:
            new_traversing_path.append(node)

    return new_traversing_path


def get_traversing_path(graph, start, end):
    eulerian_graph = get_eulerian_path(graph)
    tour = construct_tour(eulerian_graph)

    if len(tour["tour"]) == 0:
        return []

    try:
        return _get_traversing_path_helper(eulerian_graph, tour, start, end)
    except:
        breakpoint()


def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == "__main__":
    # path = [(0, 0), (0, 1), (1, 1), (2, 1), (1, 1), (2, 2), (3, 3)]

    path = [(1, 1), (2, 2), (3, 3), (5, 5), (0, 1), (1, 1)]

    # path = [
    #     (-1, -5),
    #     (0, 0),
    #     (1, 1),
    #     (1, 0),
    #     (0, 0),
    #     (5, 5),
    #     (6, 6),
    #     (7, 7),
    #     (6, 6),
    #     (3, 3),
    # ]

    graph = graph_from_path(path)

    start = (1, 1)
    end = (3, 3)

    print(get_traversing_path(graph, start, end))

    # traversing_path = get_traversing_path(eulerian_graph, start, end)
    # print(traversing_path)

    # breakpoint()

    # graph_where(eulerian_graph, start)
