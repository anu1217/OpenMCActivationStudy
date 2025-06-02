import openmc
import openmc.deplete
from pathlib import Path
import argparse
import yaml
import numpy as np
import h5py

def alara_element_densities(alara_fp):
    '''
    Creates a dictionary where keys = element names (str) and values = element density (float)
    inputs:
        alara_filepath : path to ALARA element library
    '''
    with open(alara_fp) as ALARA_Lib:
        libLines = ALARA_Lib.readlines()
    num_lines = len(libLines)
    density_dict = {}
    line_num = 0
    while line_num < num_lines:
        element_data = libLines[line_num].strip().split()
        element_name = element_data[0].lower()
        density_dict[element_name] = float(element_data[3])
        line_num += int(element_data[4]) + 1
    return density_dict
   
def make_materials(elements, density_dict):
    '''
    Creates an OpenMC Materials object using user-specified elements
    inputs:
        elements: iterable of element names (str)
        density_dict: dictionary with keys = element names (str) and values = element density (float)
    '''    
    mats = openmc.Materials([])
    for element_id, element in enumerate(elements):
        mat = openmc.Material(material_id=element_id+1, name=element)
        mat.add_element(element, 1.00)
        mat.set_density('g/cm3', density_dict.get(element.lower()))
        mats.append(mat)
    return mats

def make_spherical_shells(inner_radius, layers, outer_boundary_type):    
    '''
    Creates a set of concentric spherical shells, each with its own material & inner/outer radius.
    inputs:
        inner_radius: the radius of the innermost spherical shell
        layers: iterable of tuples of OpenMC Material object and its respective thickness (float)
    '''
    inner_sphere = openmc.Sphere(r = inner_radius)
    cells = [openmc.Cell(fill = None, region = -inner_sphere)]
    for (material, thickness) in layers:
        outer_radius = inner_radius + thickness
        outer_sphere = openmc.Sphere(r = outer_radius)
        cells.append(openmc.Cell(fill = material, region = +inner_sphere & -outer_sphere))
        inner_radius = outer_radius
        inner_sphere = outer_sphere
        material.volume = 4.0/3.0 * np.pi * ((inner_radius + thickness)**3 - (inner_radius)**3)
        inner_radius = inner_radius + thickness
    outer_sphere.boundary_type = outer_boundary_type    
    cells.append(openmc.Cell(fill = None, region = +outer_sphere))
    geometry = openmc.Geometry(cells)    
    return geometry

def make_neutron_source(energy):
    point_source = openmc.stats.Point(xyz=(0.0, 0.0, 0.0))
    energy_dist = openmc.stats.Discrete(energy, 1.0)
    neutron_source = openmc.Source(space = point_source, energy = energy_dist, strength = 1.0)
    return neutron_source

def make_settings(neutron_source, total_batches, inactive_batches, num_particles, run_mode):
    sets = openmc.Settings()
    sets.batches = total_batches
    sets.inactive = inactive_batches
    sets.particles = num_particles
    sets.source = neutron_source
    sets.run_mode = run_mode
    return sets

#Only executed if external geometry and materials are imported
def make_depletion_volumes(neutron_model, mesh_file):
    materials = neutron_model.materials
    spherical_mesh = openmc.SphericalMesh(np.arange(0, 1110, 10), origin = (0.0, 0.0, 0.0), mesh_id=2, name="spherical_mesh") 
    mat_vols = spherical_mesh.material_volumes(neutron_model, n_samples=7000000)

    total_volumes = {}
    for mat_id, volumes in mat_vols.items():
        total_volumes[mat_id] = np.sum(volumes)     
    for material in materials:
       material.volume = total_volumes[material.id]
    return neutron_model   

def deplete_wc(neutron_model, mesh_file, chain_file, timesteps, source_rates, norm_mode, timestep_units):
    spherical_mesh = openmc.SphericalMesh(np.arange(0, 1110, 10), origin = (0.0, 0.0, 0.0), mesh_id=2, name="spherical_mesh")
    
    model_nuclide_names = []
    for material in neutron_model.materials:
       material.depletable = True
       material_nuclides = material.nuclides
       for material_nuclide in material_nuclides:
            model_nuclide_names.append(material_nuclide.name)

    activations_mats_hcpb = spherical_mesh.get_homogenized_materials(neutron_model, n_samples=7000000)
    activations_mats_hcpb_object = openmc.Materials(activations_mats_hcpb)
    activations_mats_hcpb_object.export_to_xml("Activation_Materials_hcpb_sm.xml")   

    depletion_chain = openmc.deplete.Chain.from_xml(Path(chain_file).resolve())

    nuclide_names = []
    for nuclide in depletion_chain.nuclides:
        if nuclide.name in model_nuclide_names:
            nuclide_names.append(nuclide.name)
            
    fluxes, micros = openmc.deplete.get_microxs_and_flux(neutron_model, spherical_mesh, nuclide_names,
                                                          depletion_chain.reactions,
                                                          openmc.mgxs.GROUP_STRUCTURES["VITAMIN-J-175"])
    operator = openmc.deplete.IndependentOperator(openmc.Materials(activations_mats_hcpb), fluxes, micros, chain_file=chain_file, normalization_mode = norm_mode)    
    copy_timesteps = list(timesteps)
    copy_source_rates = list(source_rates)
    integrator_list = []
    for timestep, source_rate in zip(timesteps, source_rates):
        while copy_source_rates[-1] == 0:
            integrator = openmc.deplete.PredictorIntegrator(operator, copy_timesteps, source_rates = copy_source_rates, timestep_units = timestep_units)
            integrator_list.append(integrator)
            copy_timesteps.pop()
            copy_source_rates.pop()

    integrator_list.reverse()     
    #Letting integrator_list go from the fewest number of decay intervals to the most        
    for int_index, integrator in enumerate(integrator_list):
        integrator.integrate(path=f"depletion_results_hcpb_decay_set_{int_index}_sm.h5")        
    return activations_mats_hcpb, spherical_mesh, integrator_list, neutron_model

def make_photon_sources(integrator_list, activations_mats_hcpb, spherical_mesh, neutron_model, inputs):
    for int_index in range(len(integrator_list)):
        results = openmc.deplete.Results(f"depletion_results_hcpb_decay_set_{int_index}_sm.h5")
        photon_sources = np.empty(110, dtype=object)     
        activated_mats_list = set(results[-1].index_mat.keys())

        for mat_index, mat in enumerate(activations_mats_hcpb) :   
            if str(mat.id) in activated_mats_list :
                mat = results[-1].get_material(str(mat.id))
                energy = mat.get_decay_photon_energy(clip_tolerance=1e-03)
                if energy == None:
                    photon_source = openmc.IndependentSource(
                    energy = energy,
                    particle = 'photon',
                    strength = 0.0)
                else:    
                    photon_source = openmc.IndependentSource(
                    energy = energy,
                    particle = 'photon',
                    strength = energy.integral())
                
            else:
                photon_source = openmc.IndependentSource(
                energy = openmc.stats.Discrete(0, 1.0),
                particle = 'photon',
                strength = 0.0)
            
            photon_sources[mat_index] = photon_source
        photon_model = neutron_model
        photon_model.settings.source = openmc.MeshSource(spherical_mesh, photon_sources)
        photon_model.settings.export_to_xml(f'settings_{int_index}_hcpb_sm.xml')

    return photon_model, integrator_list
    
def make_photon_tallies(coeff_geom, photon_model, integrator_list):
    dose_energy, dose = openmc.data.dose_coefficients('photon', geometry=coeff_geom)
    dose_filter = openmc.EnergyFunctionFilter(dose_energy, dose)
    tally_mesh = openmc.SphericalMesh(np.arange(0, 1650, 50), origin = (0.0, 0.0, 0.0), mesh_id=3, name="tally_mesh")
    tally_mesh_filter = openmc.MeshFilter(tally_mesh)
    #Could also use an unstructured mesh filter here - spherical mesh filter for visualization purposes
    particle_filter = openmc.ParticleFilter('photon')
    energy_filter_flux = openmc.EnergyFilter.from_group_structure("VITAMIN-J-42")

    flux_tally = openmc.Tally(tally_id=1, name="photon flux tally")
    flux_tally.filters = [tally_mesh_filter, energy_filter_flux, particle_filter]
    flux_tally.scores = ['flux']

    dose_tally = openmc.Tally(tally_id=2, name="dose tally")
    dose_tally.filters = [tally_mesh_filter, dose_filter, energy_filter_flux, particle_filter]
    dose_tally.scores = ['flux']

    photon_model.tallies = [flux_tally, dose_tally]
    #Change boundary condition:
    photon_model.geometry.get_all_surfaces()[15].boundary_type="white"    

    ww_mesh = openmc.RegularMesh(mesh_id=4, name="weight_window_mesh")
    ww_mesh.dimension = (10, 10, 10)
    ww_mesh.lower_left = (-3200.0, -3200.0, -3200.0)
    ww_mesh.upper_right = (3200.0, 3200.0, 3200.0)

    wwg = openmc.WeightWindowGenerator(
    method='magic',
    mesh=ww_mesh,
    max_realizations=10
    )

    for int_index in range(len(integrator_list)):
        #Reassign settings (which contains source) for each decay step
        photon_model.settings = openmc.Settings.from_xml(f'settings_{int_index}_hcpb_sm.xml')
        photon_model.settings.weight_window_generators = wwg
        photon_model.export_to_model_xml(path=f'photon_model_hcpb_{int_index}_sm.xml')

#----------------------------------------------------------------------------------
#Define variables and execute all functions:

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OpenMC_WC_YAML', default = "R2S_HCPB.yaml", help="Path (str) to YAML containing inputs for WC_Neutron_Transport")
    parser.add_argument('--ext_mat_geom', default = True, help="Specify whether materials and geometry come from external model")
    parser.add_argument('--photon_transport', default = False, help="Run only photon transport (no depletion)")
    args = parser.parse_args()
    return args

def read_yaml(args):
    with open(args.OpenMC_WC_YAML, 'r') as transport_file:
        inputs = yaml.safe_load(transport_file)
    return inputs

def create_materials_obj(inputs):
    densities = alara_element_densities(inputs['filename_dict']['elelib_fp'])
    materials = make_materials(inputs['mat_info']['element_list'],
                        densities)
    return materials

def create_geometry_obj(materials, inputs):
    geom_info = inputs['geom_info']
    layers = zip(materials, geom_info['thicknesses'])
    geometry = make_spherical_shells(geom_info['inner_radius'],
                    layers,
                    geom_info['outer_boundary_type'])
    return geometry

def create_neutron_model(inputs, materials, geometry):
    settings_info = inputs['settings_info']
    neutron_source = make_neutron_source(inputs['particle_energy'])
    settings = make_settings(neutron_source,
                    settings_info['total_batches'],
                    settings_info['inactive_batches'],
                    settings_info['num_particles'],
                    settings_info['run_mode'])   
    neutron_model = openmc.model.Model(geometry = geometry, materials = materials, settings = settings)
    neutron_model.export_to_model_xml("neutron_model_hcpb_sm.xml")
    return neutron_model

def run_depletion(inputs, neutron_model):
    dep_params = inputs['dep_params']
   
    activations_mats_hcpb, spherical_mesh, integrator, neutron_model = deplete_wc(neutron_model,
                inputs['filename_dict']['mesh_file'],
                dep_params['chain_file'],
                dep_params['timesteps'],
                dep_params['source_rates'],
                dep_params['norm_mode'],
                dep_params['timestep_units'])
    return activations_mats_hcpb, spherical_mesh, integrator, neutron_model

def main():        
    args = parse_args()
    inputs = read_yaml(args)

    openmc.config['chain_file'] = inputs['dep_params']['chain_file']
    if args.ext_mat_geom == True : #Import materials and geometry from external model
        ext_model = openmc.model.Model.from_model_xml(inputs['filename_dict']['ext_model'])
        materials = ext_model.materials
        geometry = ext_model.geometry        
    else:    
        materials = create_materials_obj(inputs)
        geometry = create_geometry_obj(materials, inputs)

    neutron_model = create_neutron_model(inputs, materials, geometry)

    #Set to True to run photon transport only (no depletion):    
    if args.photon_transport == True:
        activations_mats_hcpb = openmc.Materials.from_xml("Activation_Materials_hcpb_sm.xml")
        spherical_mesh = openmc.SphericalMesh(np.arange(0, 1110, 10), origin = (0.0, 0.0, 0.0), mesh_id=2, name="spherical_mesh")
        integrator_list = ['int_0','int_1']
    else:
        if args.ext_mat_geom == True :
          neutron_model = make_depletion_volumes(neutron_model, inputs['filename_dict']['mesh_file'])
        activations_mats_hcpb, spherical_mesh, integrator_list, neutron_model = run_depletion(inputs, neutron_model)

    
    photon_model, integrator_list = make_photon_sources(integrator_list, activations_mats_hcpb, spherical_mesh, neutron_model, inputs)
    photon_tallies = make_photon_tallies(inputs['coeff_geom'], photon_model, integrator_list)
   
if __name__ == "__main__":
    main()