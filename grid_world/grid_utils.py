import pandas as pd
import numpy as np
from utils import utils
import random
import math


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

def GetPathCorList(grid,point1,point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    cor_list = []
    while x1 != x2 or y1 != y2:
        cor_list.append((x1,y1))
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
                cor_list.append((x1-sx,y1))
            else:
                cor_list.append((x1,y1-sy))
    
    cor_list.append((x1,y1))
    return cor_list

def StatesToStateActionPairs(states):
    '''
    actions: 0:stay,1:up,2:down,3:left,4:right
    '''
    pairs = []
    for i in range(len(states)-1):
        if states[i] == states[i+1]:
            pairs.append([states[i],0])
        elif states[i][0] == states[i+1][0]:
            if states[i][1] == states[i+1][1] + 1:
                pairs.append([states[i],2])
            elif states[i][1] == states[i+1][1] - 1:
                pairs.append([states[i],1])
        elif states[i][1] == states[i+1][1]:
            if states[i][0] == states[i+1][0] + 1:
                pairs.append([states[i],3])
            elif states[i][0] == states[i+1][0] - 1:
                pairs.append([states[i],4])
        
    return pairs

def GetFeature(env_array,i,j):
    feature = 0
    for row in range(0,env_array.shape[0]):
        for col in range(0,env_array.shape[1]):
            if env_array[row,col] != 0:
                dis = (row-i)**2+(col-j)**2
                if dis == 0:
                    dis = 0.5
                dis = math.sqrt(dis)
                feature += env_array[row,col]/dis
    return feature

def CoordToState(coord,width):
        x,y = coord
        return int(y*width+x)
    
def StateToCoord(state,width):
    x = state%width
    y = state//width
    return (x,y)
    