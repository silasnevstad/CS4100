import gridgame as gg
import numpy as np
import time

########################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

########################################################################################################################

gg.setup(GUI=False, render_delay_sec=0.1, gs=7)

########################################################################################################################

# Initialization

# grid represents the current state of the board. 

# -1 indicates an empty cell
# 0 indicates a cell colored in the first color (indigo by default)
# 1 indicates a cell colored in the second color (taupe by default)
# 2 indicates a cell colored in the third color (veridian by default)
# 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.

# Each shape is represented as a list containing three elements: a) the brush type (number between 0-8),
# b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between
# 0-3)

# For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top
# left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

########################################################################################################################

grid, placedShapes, done = gg.execute('export')
# input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to
# True. Uncomment to enable.
print(grid, placedShapes, done)

####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.


##########################################
# Write all your code in the area below. 
##########################################


def is_valid_placement(grid, shape, position, color):
    """
    Check if a shape can be placed at a given position on the grid. Loops through each cell in the shape and checks if
    it is within the bounds of the grid and if it is empty. Also checks if the shape is adjacent to a cell of the same
    color.

    :param grid: Current state of the grid
    :param shape: Shape to be placed
    :param position: Position to place the shape
    :param color: Color of the shape
    :return: True if the shape can be placed, False otherwise
    """
    # Loop through each cell in the shape
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                x, y = position[0] + j, position[1] + i
                # Check if the cell is within the bounds of the grid and if it is empty
                if x >= gg.gridSize or y >= gg.gridSize:
                    return False
                if grid[y, x] != -1:
                    return False
                # Check if the cell is adjacent to a cell of the same color
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gg.gridSize and 0 <= ny < gg.gridSize and grid[ny, nx] == color:
                        return False
    return True


def get_empty_cells(grid):
    """
    Get a list of all empty cells on the grid.

    :param grid: Current state of the grid
    :return: List of empty cells
    """
    return [(y, x) for y in range(gg.gridSize) for x in range(gg.gridSize) if grid[y, x] == -1]


def most_constrained_variable(grid, domains):
    """
    Get the cell with the fewest remaining values in its domain.

    :param grid: Current state of the grid
    :param domains: Current domains of each cell
    :return: Cell with the fewest remaining values in its domain
    """
    empty_cells = get_empty_cells(grid)
    return min(empty_cells, key=lambda cell: len(domains[cell[0]][cell[1]]))


def least_constraining_value(grid, domains, cell):
    """
    Get the colors that are least constraining for a given cell. The least constraining colors are the ones that are
    adjacent to the fewest number of empty cells. This is done by sorting the colors by the number of empty cells that
    they are adjacent to.

    :param grid: Current state of the grid
    :param domains: Current domains of each cell
    :param cell: Cell to get the least constraining colors for
    :return: List of colors sorted by the number of empty cells that they are adjacent to
    """
    y, x = cell
    colors = list(domains[y][x])
    return sorted(colors, key=lambda color: sum(1 for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                                if 0 <= y + dy < gg.gridSize and 0 <= x + dx < gg.gridSize
                                                and grid[y + dy][x + dx] == -1 and color in domains[y + dy][x + dx]))


def forward_checking(grid, domains, shape, position, color):
    """
    Perform forward checking to update the domains of the cells adjacent to the placed shape. This is done by removing
    the color of the placed shape from the domains of the adjacent cells. If the domain of an adjacent cell becomes
    empty, the placement is invalid and None is returned.

    :param grid: Current state of the grid
    :param domains: Current domains of each cell
    :param shape: Shape that was placed
    :param position: Position where the shape was placed
    :param color: Color of the shape
    :return: Updated domains or None if the placement is invalid
    """
    new_domains = [row[:] for row in domains]  # Copy domains
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                y, x = position[1] + i, position[0] + j
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gg.gridSize and 0 <= nx < gg.gridSize and grid[ny][nx] == -1:
                        new_domains[ny][nx] = new_domains[ny][nx] - {color}
                        if not new_domains[ny][nx]:
                            return None
    return new_domains


def shape_score(shape):
    """
    Calculate a score for a shape based on the number of cells it covers.

    :param shape: Shape to calculate the score for
    :return: Score of the shape
    """
    return np.sum(shape)


def backtrack(domains, depth=0):
    """
    Perform backtracking search to find a solution to the map coloring problem. The search is done by selecting the most
    constrained cell and trying to place the most constraining shape at that cell. If a placement is invalid, the search
    backtracks and tries the next shape. If a placement is valid, forward checking is performed to update the domains
    of the adjacent cells. If the domains become empty, the placement is invalid and the search backtracks. If all cells
    are filled, the search returns the grid and the shapes used to color it.

    :param domains: Domains of each cell
    :param depth: Current depth of the search
    :return: Grid if a solution is found, None otherwise
    """
    global grid, placedShapes, done
    # If all cells are filled, return the grid
    if len(get_empty_cells(grid)) == 0:
        return grid

    # Select the most constrained cell (cell with the fewest remaining values in its domain)
    cell = most_constrained_variable(grid, domains)
    y, x = cell

    # Sort shapes by score (prefer larger shapes)
    sorted_shapes = sorted(enumerate(gg.shapes), key=lambda x: shape_score(x[1]), reverse=True)

    # Try to place the most constraining shape at the cell
    for shape_index, shape in sorted_shapes:
        # Try to place the shape with the least constraining value
        for color in least_constraining_value(grid, domains, cell):
            # If the placement is valid, place the shape and update the domains
            if is_valid_placement(grid, shape, (x, y), color):
                execute_shape_placement(shape_index, (x, y), color)

                # Perform forward checking, if the placement is invalid, backtrack
                new_domains = forward_checking(grid, domains, shape, (x, y), color)
                if new_domains:
                    result = backtrack(new_domains, depth + 1)
                    if result is not None:
                        return result

                # Use execute function to undo the shape placement
                grid, placedShapes, done = gg.execute('undo')

    return None


def execute_shape_placement(shape_index, position, color):
    """
    Execute the placement of a shape on the grid. This is done by switching to the correct shape, color and position and
    then placing the shape.

    :param shape_index: Index of the shape to place
    :param position: Position to place the shape
    :param color: Color of the shape
    :return: Grid, placed shapes and done flag after placing the shape
    """
    global grid, placedShapes, done
    # Switch to the correct shape
    while gg.currentShapeIndex != shape_index:
        grid, placedShapes, done = gg.execute('switchshape')

    # Switch to the correct color
    while gg.currentColorIndex != color:
        grid, placedShapes, done = gg.execute('switchcolor')

    # Move to the correct position
    while gg.shapePos[0] < position[0]:
        grid, placedShapes, done = gg.execute('right')
    while gg.shapePos[0] > position[0]:
        grid, placedShapes, done = gg.execute('left')
    while gg.shapePos[1] < position[1]:
        grid, placedShapes, done = gg.execute('down')
    while gg.shapePos[1] > position[1]:
        grid, placedShapes, done = gg.execute('up')

    # Place the shape
    grid, placedShapes, done = gg.execute('place')
    return grid, placedShapes, done

def solve_map_coloring():
    """
    Solve the map coloring problem by performing backtracking search.

    :return: None
    """
    global grid, placedShapes, done
    domains = [[set(range(4)) for _ in range(gg.gridSize)] for _ in range(gg.gridSize)]
    backtrack(domains)


# Main execution
solve_map_coloring()

print(grid, placedShapes, done)

########################################

# Do not modify any of the code below.

########################################

end = time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end - start))