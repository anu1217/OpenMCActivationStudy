import openmc

def make_spherical_shells(materials, thicknesses, inner_radius):    
    '''
    Creates a set of concentric spherical shells, each with its own material & inner/outer radius.
    inputs:
        materials : iterable of OpenMC Material objects
        thicknesses: thickness of each OpenMC Material
        inner_radius: the radius of the innermost spherical shell
    '''
    layers = zip(materials, thicknesses)
    inner_sphere = openmc.Sphere(r = inner_radius)
    cells = [openmc.Cell(fill = None, region = -inner_sphere)]
    for (material, thickness) in layers:
        outer_radius = inner_radius + thickness
        outer_sphere = openmc.Sphere(r = outer_radius)
        cells.append(openmc.Cell(fill = material, region = +inner_sphere & -outer_sphere))
        outer_radius = inner_radius
        outer_sphere = inner_sphere
    outer_sphere.boundary_type = 'vacuum'     
    cells.append(openmc.Cell(fill = None, region = +outer_sphere)) 
    geometry = openmc.Geometry(cells)    
    return geometry