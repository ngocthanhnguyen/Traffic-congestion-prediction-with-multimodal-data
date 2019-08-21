import sys
sys.path.append('../')
from jpmesh.jpmesh import QuarterMesh, Coordinate, Angle

WD = {
    'input': {
        'extract_raster': '/home/tnnguyen/Thesis/bigdata2019/Final-Thesis/dataset/output/'
    },
    'output': {
        'extract_raster': '/home/tnnguyen/Thesis/bigdata2019/Final-Thesis/dataset/raster_imgs/'
    }
}

ACCIDENT_TYPES = ('0', '{v_to_m}', '{v_to_v_encounter}', '{v_to_v_front}', '{vehicle_own}',
                  '{v_to_v_left}', '{v_to_v_other}', '{v_to_v_rear}', '{v_to_v_right}')

RASTER_CONFIG = {
    # factor map
    'width'             : 1060,
    'height'            : 1060,
    'offset_lat'        : 0.002083333,
    'offset_long'       : 0.003125,
    'start_lat'         : 33.584375,
    'start_long'        : 134.0015625,
    'pixel_area'        : 0.25 ** 2, # 0.25km
    
    # factor channel index
    'congestion_channel'        : 0,
    'rainfall_channel'          : 1,
    'accident_channel'          : 2,

    'num_factors'               : 3
}
