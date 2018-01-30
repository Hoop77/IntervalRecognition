from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

def spline(y0, y1, length):
    x = [0, 1, 2, 3, 4, 5, 6]
    y2 = min(y0, y1) + np.abs(y1 - y0) / 2.
    y = [y1, y2, y0, y2, y1, y2, y0]
    tck = interpolate.splrep(x, y)
    xnew = np.linspace(2, 4, length, True)
    return interpolate.splev(xnew, tck)

def interpolate_points(frequencies, result, idx, x0, x1, y0, y1):
    # if two consecutive points equal in y-value there is no need to interpolate
    # we just draw a horizontal line until x1
    if y0 == y1:
        while idx < len(frequencies) and frequencies[idx] <= x1:
            result[idx] = y0
            idx += 1
    else:
        start_idx = idx
        while idx < len(frequencies) and frequencies[idx] <= x1:
            idx += 1
        
        result[start_idx:idx] = spline(y0, y1, idx - start_idx)

    return idx

def eq(frequencies, points, spline="cubic"):
    result = np.ones(len(frequencies))
    
    max_freq = frequencies[-1]

    # keep only points within the frequency range
    points = {x: y for x, y in points.items() if x >= 0. and x <= max_freq}

    # add border points so we have at least two points
    if 0. not in points:
        points[0.] = 1.0
    if max_freq not in points:
        points[max_freq] = 1.0

    xs = list(points.keys())
    xs.sort()

    # index to fill result with values
    idx = 0

    for i in range(1, len(xs)):
        x0 = xs[i-1]
        x1 = xs[i]
        y0 = points[x0]
        y1 = points[x1]
        idx = interpolate_points(frequencies, result, idx, x0, x1, y0, y1)

    return result