import pygame, math, sys, my_functions
from pygame.locals import *
from math import *
import numpy as np

def mapX(x, a, b, c, d):
    val = (x - a) / ((b - a) / (d - c)) + c

    return val

def multiply_normalize(a, b):
    a = np.dot(a, b)

    if a[3] != 0:
        a[0] /= a[3]
        a[1] /= a[3]
        a[2] /= a[3]

    return np.array([a[0], a[1], a[2]])


def create_vector(a, b):
    vector = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return vector

def normalize(a):
    l = sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    a = np.array([a[0] / l, a[1] / l, a[2] / l])
    return a

def rotate_Z(theta):
    return np.array([[cos(theta / 2), sin(theta / 2), 0, 0],
                     [-sin(theta / 2), cos(theta / 2), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rotate_Y(theta):
    return np.array([[cos(theta / 2), 0, sin(theta / 2), 0],
                     [0, 1, 0, 0],
                     [-sin(theta / 2), 0, cos(theta / 2), 0],
                     [0, 0, 0, 1]])

def rotate_Y_3(theta):
    return np.array([[cos(theta / 2), 0, sin(theta / 2)],
                     [0, 1, 0],
                     [-sin(theta / 2), 0, cos(theta / 2)]])

def rotate_X(theta):
    return np.array([[1, 0, 0, 0],
                     [0, cos(theta / 2), sin(theta / 2), 0],
                     [0, -sin(theta / 2), cos(theta / 2), 0],
                     [0, 0, 0, 1]])


def matrix_pointat(pos, fw, up):
    newForward = np.subtract(fw, pos)
    newForward = normalize(newForward)

    a = np.dot(newForward, np.dot(newForward, up))
    newUp = np.subtract(up, a)
    newUp = normalize(newUp)

    newRight = np.cross(newUp, newForward)

    point_at = np.array([[  newRight[0],   newRight[1],   newRight[2], 0],
                         [     newUp[0],      newUp[1],      newUp[2], 0],
                         [newForward[0], newForward[1], newForward[2], 0],
                         [       pos[0],        pos[1],        pos[2], 1]])

    return point_at

def matrix_inverse(m):
    inverse = np.array([[m[0, 0], m[1, 0], m[2, 0], 0],
                        [m[0, 1], m[1, 1], m[2, 1], 0],
                        [m[0, 2], m[1, 2], m[2, 2], 0],
                        [-(m[3, 0] * m[0, 0] + m[3, 1] * m[0, 1] + m[3, 2] * m[0, 2]),
                         -(m[3, 0] * m[1, 0] + m[3, 1] * m[1, 1] + m[3, 2] * m[1, 2]),
                         -(m[3, 0] * m[2, 0] + m[3, 1] * m[2, 1] + m[3, 2] * m[2, 2]),
                         1]])

    return inverse