import openmc
import openmc.deplete
import numpy as np

model = openmc.model.Model.from_model_xml("neutron_model_pyne.xml")

for material in model.materials:
    material.depletable = True


spherical_mesh = openmc.SphericalMesh(np.arange(995, 1010, 5), origin = (0.0, 0.0, 0.0), mesh_id=1, name="spherical_mesh")
activation_mats_spherical = spherical_mesh.get_homogenized_materials(model, n_samples=700000, include_void = False)
activation_mats_object_spherical = openmc.Materials(activation_mats_spherical)
activation_mats_object_spherical.export_to_xml("Activation_Materials_spherical.xml") # looks ok

unstructured_mesh = openmc.UnstructuredMesh('Mesh.h5', mesh_id = 2, library='moab') 
activation_mats_unstructured = unstructured_mesh.get_homogenized_materials(model, n_samples=700000, include_void = False)
activation_mats_object_unstructured = openmc.Materials(activation_mats_unstructured)
activation_mats_object_unstructured.export_to_xml("Activation_Materials_unstructured.xml") # makes too many void materials
