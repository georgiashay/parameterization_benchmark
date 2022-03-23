import numpy as np

def get_area_distortion(uv_areas, mesh_areas):
    area_distortions = uv_areas/mesh_areas + mesh_areas/uv_areas
    max_area_distortion = np.max(area_distortions) - 2
    
    area_errors = np.abs(uv_areas - mesh_areas)
    average_area_error = np.sum(area_errors)
    
    return area_distortions, area_errors, max_area_distortion, average_area_error