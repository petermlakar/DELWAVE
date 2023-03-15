import numpy as np

def get_spatial_map(j, i, t):

    map = np.zeros(shape = (90, 89), dtype = np.float32)

    for x in range(0, map.shape[0]):
        for y in range(0, map.shape[1]):
            map[x, y] = np.exp(-0.5*(np.square(x - j) + np.square(y - i))/t)/(np.sqrt(2*np.pi*t))

    return (map - map.min())/(map.max() - map.min())
    
def spatial_map(name, t = 20):

    if name == "AA":
        return get_spatial_map(74,  5, t)
    if name == "MB":
        return get_spatial_map(17, 59, t)
    if name == "GD":
        return get_spatial_map(79, 13, t)
    if name == "OB":
        return get_spatial_map(34, 27, t)
    if name == "OB2":
        return get_spatial_map(44, 34, t)
    if name == "OB3":
        return get_spatial_map(63, 39, t)

