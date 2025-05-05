import random

def read_graph(filename):
    vertices = set()
    edges = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Expect format: u v w
                parts = line.split()
                if len(parts) != 3:
                    print(f"Skipping invalid line: {line}")
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    w = int(parts[2])
                    vertices.add(u)
                    vertices.add(v)
                    edges.add((min(u, v), max(u, v)))  # Store undirected edge
                except ValueError:
                    print(f"Skipping line with non-integer values: {line}")
                    continue
        return vertices, edges
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None, None

def generate_updates(vertices, edges, num_updates):
    updates = []
    min_vertex = min(vertices)
    max_vertex = max(vertices)

    for _ in range(num_updates):
        # Randomly choose two vertices
        u = random.randint(min_vertex, max_vertex)
        v = random.randint(min_vertex, max_vertex)
        while u == v:  # Ensure u and v are different
            v = random.randint(min_vertex, max_vertex)

        # Ensure u < v for consistency in output
        u, v = min(u, v), max(u, v)

        # Randomly choose operation: del or ins
        if random.random() < 0.5:  # 50% chance for deletion
            updates.append(f"{u} {v} del")
        else:  # 50% chance for insertion
            weight = random.randint(1, 1000)  # Random weight between 1 and 1000
            updates.append(f"{u} {v} ins {weight}")

    return updates

def write_updates(updates, filename="updates4000.txt"):
    with open(filename, 'w') as f:
        for update in updates:
            f.write(update + '\n')
    print(f"Updates written to {filename}")

def main():
    # Get graph file from user
    graph_file = input("Enter the path to the graph file: ")
    vertices, edges = read_graph(graph_file)
    
    if vertices is None or not vertices:
        print("Failed to read graph. Exiting.")
        return

    print(f"Graph loaded: {len(vertices)} vertices, {len(edges)} edges")

    # Get number of updates from user
    try:
        num_updates = int(input("Enter the number of updates to generate: "))
        if num_updates <= 0:
            print("Number of updates must be positive.")
            return
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
        return

    # Generate updates
    updates = generate_updates(vertices, edges, num_updates)
    
    # Write updates to file
    write_updates(updates)

if __name__ == "__main__":
    main()