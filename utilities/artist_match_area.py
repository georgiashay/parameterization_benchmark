import numpy as np

def get_artist_area_match(mesh_areas, uv_areas_artist, uv_areas_measure):
    my_area_diffs = np.abs(uv_areas_measure - mesh_areas)
    artist_area_diffs = np.abs(uv_areas_artist - mesh_areas)
    
    artist_me_diff = my_area_diffs - artist_area_diffs
    artist_me_diff[np.where(artist_me_diff < 0)] = 0
    
    max_diff = np.max(artist_me_diff)
    total_diff = np.sum(artist_me_diff)
    
    return artist_me_diff, max_diff, total_diff