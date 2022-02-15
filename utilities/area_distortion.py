import numpy as np

def get_area_distortion(uv_areas, mesh_areas):
    area_distortions = np.abs(uv_areas - mesh_areas)
    max_area_distortion = np.max(area_distortions)
    total_area_distortion = np.sum(area_distortions)
    
    return area_distortions, max_area_distortion, total_area_distortion
            