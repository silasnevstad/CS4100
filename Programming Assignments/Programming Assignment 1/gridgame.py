import time
import pygame
import numpy as np

# Constants, overridden by setup()
gridSize = 6
cellSize = 40
screenSize = gridSize * cellSize
fps = 60
sleeptime = 0.1

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Colors for the shapes
colors = ['#988BD0', '#504136', '#457F6E', '#F7C59F']  # Indigo, Taupe, Viridian, Peach

colorIdxToName = {0: "Indigo", 1: "Taupe", 2: "Viridian", 3: "Peach"}

# Shapes
shapes = [
    np.array([[1]]),  # 1x1 square
    np.array([[1, 0], [0, 1]]),  # 2x2 square with holes
    np.array([[0, 1], [1, 0]]),  # 2x2 square with holes
    np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),  # 2x4 rectangle with holes
    np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),  # 2x4 rectangles with holes
    np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),  # 4x2 rectangle with holes
    np.array([[0, 1, 0, 1], [1, 0, 1, 0]]),  # 4x2 rectangles with holes
    np.array([[0, 1, 0], [1, 0, 1]]),  # T shape with holes
    np.array([[1, 0, 1], [0, 1, 0]])  # T shape with holes
]

shapesIdxToName = {
    0: "Square",
    1: "SquareWithHoles",
    2: "SquareWithHolesTranspose",
    3: "RectangleWithHoles",
    4: "RectangleWithHolesTranspose",
    5: "RectangleVerticalWithHoles",
    6: "RectangleVerticalWithHolesTranspose",
    7: "SparseTShape",
    8: "SparseTShapeReverse",
}

# Global variables
screen = None
clock = None
grid = None
currentShapeIndex = None
currentColorIndex = None
shapePos = None
placedShapes = None


def drawGrid(screen):
    for x in range(0, screenSize, cellSize):
        for y in range(0, screenSize, cellSize):
            rect = pygame.Rect(x, y, cellSize, cellSize)
            pygame.draw.rect(screen, black, rect, 1)


def drawShape(screen, shape, color, pos):
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                rect = pygame.Rect((pos[0] + j) * cellSize, (pos[1] + i) * cellSize, cellSize, cellSize)
                pygame.draw.rect(screen, color, rect, width=6)


def canPlace(grid, shape, pos):
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                if pos[0] + j >= gridSize or pos[1] + i >= gridSize:
                    return False
                if grid[pos[1] + i, pos[0] + j] != -1:
                    return False
    return True


def placeShape(grid, shape, pos, colorIndex):
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                grid[pos[1] + i, pos[0] + j] = colorIndex


def removeShape(grid, shape, pos):
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                grid[pos[1] + i, pos[0] + j] = -1


def checkGrid(grid):
    if -1 in grid:
        return False

    for i in range(gridSize):
        for j in range(gridSize):
            color = grid[i, j]
            if i > 0 and grid[i - 1, j] == color:
                return False
            if i < gridSize - 1 and grid[i + 1, j] == color:
                return False
            if j > 0 and grid[i, j - 1] == color:
                return False
            if j < gridSize - 1 and grid[i, j + 1] == color:
                return False
    return True


def exportGridState(grid):
    return grid
    # return ''.join(chr(cell + 65) for row in grid for cell in row)


def importGridState(gridState):
    grid = np.array([ord(char) - 65 for char in gridState]).reshape((gridSize, gridSize))
    return grid

def refresh():
    global screen, gridSize, grid, cellSize, colors, currentColorIndex, currentShapeIndex, shapePos, shapes, sleeptime
    screen.fill(white)
    drawGrid(screen)

    for i in range(gridSize):
        for j in range(gridSize):
            if grid[i, j] != -1:
                rect = pygame.Rect(j * cellSize, i * cellSize, cellSize, cellSize)
                pygame.draw.rect(screen, colors[grid[i, j]], rect)

    drawShape(screen, shapes[currentShapeIndex], colors[currentColorIndex], shapePos)

    pygame.display.flip()
    clock.tick(fps)
    time.sleep(sleeptime)



def setup(GUI=True, render_delay_sec=0.1, gs=6):
    global gridSize, screen, clock, grid, currentShapeIndex, currentColorIndex, shapePos, placedShapes, sleeptime, screenSize
    gridSize = gs
    sleeptime = render_delay_sec
    grid = np.full((gridSize, gridSize), -1)
    currentShapeIndex = 0
    currentColorIndex = 0
    shapePos = [0, 0]
    placedShapes = []

    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((screenSize, screenSize))
        pygame.display.set_caption("Shape Placement Grid")
        clock = pygame.time.Clock()

        refresh()
        

def loop_gui():
    global currentShapeIndex, currentColorIndex, shapePos, grid, placedShapes
    running = True
    while running:
        screen.fill(white)
        drawGrid(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    shapePos[1] = max(0, shapePos[1] - 1)
                elif event.key == pygame.K_s:
                    shapePos[1] = min(gridSize - len(shapes[currentShapeIndex]), shapePos[1] + 1)
                elif event.key == pygame.K_a:
                    shapePos[0] = max(0, shapePos[0] - 1)
                elif event.key == pygame.K_d:
                    shapePos[0] = min(gridSize - len(shapes[currentShapeIndex][0]), shapePos[0] + 1)
                elif event.key == pygame.K_p:
                    if canPlace(grid, shapes[currentShapeIndex], shapePos):
                        placeShape(grid, shapes[currentShapeIndex], shapePos, currentColorIndex)
                        placedShapes.append((currentShapeIndex, shapePos.copy(), currentColorIndex))
                        if checkGrid(grid):
                            print("All cells are covered with no overlaps and no adjacent same colors!")
                        else:
                            print("Grid conditions not met!")
                elif event.key == pygame.K_h:
                    currentShapeIndex = (currentShapeIndex + 1) % len(shapes)
                    print("Current shape", shapesIdxToName[currentShapeIndex])
                elif event.key == pygame.K_k:
                    currentColorIndex = (currentColorIndex + 1) % len(colors)
                elif event.key == pygame.K_u:  # Undo the last placed shape
                    if placedShapes:
                        lastShapeIndex, lastShapePos, lastColorIndex = placedShapes.pop()
                        removeShape(grid, shapes[lastShapeIndex], lastShapePos)
                elif event.key == pygame.K_e:  # Export the grid state
                    gridState = exportGridState(grid)
                    print("Exported Grid State:", gridState)
                    print("Placed Shapes:", placedShapes)
                elif event.key == pygame.K_i:  # Import the grid state, not needed for us.
                    # Dummy grid state for testing
                    dummyGridState = exportGridState(np.random.randint(-1, 4, size=(gridSize, gridSize)))
                    grid = importGridState(dummyGridState)
                    placedShapes.clear()  # Clear history since we are importing a new state

        # Draw already placed shapes
        for i in range(gridSize):
            for j in range(gridSize):
                if grid[i, j] != -1:
                    rect = pygame.Rect(j * cellSize, i * cellSize, cellSize, cellSize)
                    pygame.draw.rect(screen, colors[grid[i, j]], rect)

        drawShape(screen, shapes[currentShapeIndex], colors[currentColorIndex], shapePos)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def execute(command='e'):
    global currentShapeIndex, currentColorIndex, shapePos, grid, placedShapes
    #print("Enter commands (W/A/S/D to move, P to place, Q to quit, U to undo, H to change shape, K to change color):")
    # running = True
    done = False
    if command == 'E' or command == 'e' or command=='export':
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='e', key=ord('e'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
        return grid, placedShapes, done
    # while running:
    #     command = input("Command: ").strip().upper()
    #     if command == 'Q':
    #         running = False
    if command == 'W' or command == 'w' or command.lower() == "up":
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='w', key=ord('w'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
        shapePos[1] = max(0, shapePos[1] - 1)
    elif command == 'S' or command == 's' or command.lower() == "down":
        shapePos[1] = min(gridSize - len(shapes[currentShapeIndex]), shapePos[1] + 1)
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='s', key=ord('s'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
    elif command == 'A' or command == 'a' or command.lower() == "left":
        shapePos[0] = max(0, shapePos[0] - 1)
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='a', key=ord('a'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
    elif command == 'D' or command == 'd' or command.lower() == "right":
        shapePos[0] = min(gridSize - len(shapes[currentShapeIndex][0]), shapePos[0] + 1)
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='d', key=ord('d'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
    elif command == 'P' or command == 'p' or command.lower() == "place":
        if canPlace(grid, shapes[currentShapeIndex], shapePos):
            placeShape(grid, shapes[currentShapeIndex], shapePos, currentColorIndex)
            placedShapes.append((currentShapeIndex, shapePos.copy(), currentColorIndex))
            exportGridState(grid)
            new_event = pygame.event.Event(pygame.KEYDOWN, unicode='p', key=ord('p'))
            try:
                pygame.event.post(new_event)
                refresh()
            except:
                pass
            if checkGrid(grid):
                #print("All cells are covered with no overlaps and no adjacent same colors!")
                done = True
            else:
                #print("Grid conditions not met!")
                done = False
    elif command == 'H' or command == 'h' or command.lower() == "switchshape":
        currentShapeIndex = (currentShapeIndex + 1) % len(shapes)
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='h', key=ord('h'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
        #print("Current shape", shapesIdxToName[currentShapeIndex])
    elif command == 'K' or command == 'k' or command.lower() == "switchcolor":
        currentColorIndex = (currentColorIndex + 1) % len(colors)
        new_event = pygame.event.Event(pygame.KEYDOWN, unicode='k', key=ord('k'))
        try:
            pygame.event.post(new_event)
            refresh()
        except:
            pass
    elif command == 'U' or command == 'u' or command.lower() == "undo":  # Undo the last placed shape
        if placedShapes:
            lastShapeIndex, lastShapePos, lastColorIndex = placedShapes.pop()
            removeShape(grid, shapes[lastShapeIndex], lastShapePos)
            new_event = pygame.event.Event(pygame.KEYDOWN, unicode='u', key=ord('u'))
            try:
                pygame.event.post(new_event)
                refresh()
            except:
                pass

    # Display grid state
    return grid, placedShapes, done


def printGridState(grid):
    for row in grid:
        print(' '.join(f'{cell:2}' for cell in row))
    print()


def main():
    #print("Select mode: 1 for GUI, 2 for Terminal")
    #mode = input("Mode: ").strip()
    setup(True, render_delay_sec=0.1, gs=6)
    loop_gui()
    # if mode == '1':
    #     loop_gui()
    # elif mode == '2':
    #     execute()


def printControls():
    print("W/A/S/D to move the shapes.")
    print("H to change the shape.")
    print("K to change the color.")
    print("P to place the shape.")
    print("U to undo the last placed shape.")
    print("E to print the grid state from GUI to terminal.")
    print("I to import a dummy grid state.")
    print("Q to quit (terminal mode only).")
    print("Press any key to continue")

if __name__ == "__main__":
    printControls()
    main()
