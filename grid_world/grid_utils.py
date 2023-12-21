import pandas as pd
import numpy as np
from utils import utils
import random


def DrawPathOnGrid(grid,point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        grid[y1][x1] += 1
        e2 = 2 * err

        movex = False
        movey = False
        if e2 > -dy:
            err -= dy
            x1 += sx
            movex = True
        if e2 < dx:
            err += dx
            y1 += sy
            movey = True

        if movex and movey:
            if random.random() <0.5:
                grid[y1][x1 - sx] += 1
            else:
                grid[y1-sy][x1] += 1
    
    grid[y1][x1] += 1
    return grid