## Importing the libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from config import *
from PIL import Image
import sys
sys.path.append('../')
from data_utils.DataParser import *
from data_utils.DataReader import *
from data_utils.DataScaler import *
from jpmesh.jpmesh import *

def parse_data(dp, data, col_conf, target_col):
    # datetime
    data = dp.removeDelimiters(data, col_conf.index('datetime'), ('-', ' ', ':', '+09'))    
    data = dp.convertInt(data, col_conf.index('datetime'))

    # list factors    
    data = dp.countElements(data, col_conf.index('numsegments'), ',')

    # integer factors
    data = dp.convertInt(data, col_conf.index('meshcode'))
    data = dp.convertInt(data, col_conf.index('rainfall'))
    data = dp.convertInt(data, col_conf.index('congestion'))

    # categorical accident
    data = dp.convertTextType(data, col_conf.index('accident'), ACCIDENT_TYPES)
    
    return data

def convert_idx2time(id, step=5):
    hh = mm = ss = 0
    step_per_min = int(60/step)

    if id < step_per_min:
        mm = id * step
    else:
        hh = id // step_per_min
        mm = id % step_per_min * 5
    
    time = '{:02d}{:02d}'.format(hh, mm)
    return time

def dump_factor(path, factor_map):
    np.savez_compressed(path, factor_map)

def generate_factor_map(path, data, num_factors, col_conf, factor_config):
    loc = {}

    date = data[0, col_conf.index('datetime')]
    date //= 1000000

    timecode = 0
    j = 0

    timecodesPerDay = 288

    while timecode < timecodesPerDay:
        time = convert_idx2time(timecode)
        starting_time = str(date) + time
        factor_map = np.zeros((factor_config['width'], factor_config['height'],num_factors))
        
        while j < data.shape[0]:
            ending_time = data[j, col_conf.index('datetime')] // 100
            if str(ending_time) > starting_time:
                break
            elif str(ending_time) < starting_time:
                continue

            # get center coordination of each mesh
            meshcode = data[j, col_conf.index('meshcode')]
            mesh = parse_mesh_code(str(meshcode))
            mesh_center = mesh.south_west + (mesh.size / 2.0)
            latitude = mesh_center.lat.degree
            longitude = mesh_center.lon.degree

            # calculate relative positive on raster image
            loc['x'] = int((latitude - factor_config['start_lat']) // factor_config['offset_lat'])
            loc['y'] = int((longitude - factor_config['start_long']) // factor_config['offset_long'])
            if loc['x'] >= RASTER_CONFIG['width'] or loc['y'] >= RASTER_CONFIG['height']:
                continue
            
            # assign sensing data to corresponding location on raster image
            congestion = data[j, col_conf.index('congestion')] * data[j, col_conf.index('numsegments')]
            rainfall = data[j, col_conf.index('rainfall')]
            accident = data[j, col_conf.index('accident')]
            if accident > 0:
                accident = 1

            factor_map[loc['x'] ,loc['y'],factor_config['congestion_channel']]  = congestion
            factor_map[loc['x'] ,loc['y'],factor_config['rainfall_channel']]    = rainfall
            factor_map[loc['x'] ,loc['y'],factor_config['accident_channel']]    = accident

            j += 1

        print('Generating raster image for', starting_time)
        dump_factor(WD['output']['extract_raster'] + starting_time, factor_map)
        del factor_map

        timecode += 1

if __name__ == "__main__":
    # ============================================ #
    # Needed columns for extraction
    col_name = ['datetime', 'meshcode', 'rainfall', 'numsegments', 'congestion', 'accident']    
    col_idx  = [1         , 2         , 5         , 6            , 9           , 15        ]
    target_col = len(col_name) - 1

    # ============================================ #
    # Data location
    data_path = WD['input']['extract_raster']           # path to time series dataset
    output_raster_path = WD['output']['extract_raster'] # path to raster image storage

    # ============================================ #
    # Read data
    data_files = os.listdir(data_path)
    data_files.sort()
    for i in range(len(data_files)):
        data_files[i] = data_path + data_files[i]
    
    dr = DataReader(data_files, col_idx)
    ds = DataScaler()
    dp = DataParser()

    if len(sys.argv) > 2:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    else:
        start = 0
        end = len(data_files)

    for file_id in range(start, end):
        dr_tmp = DataReader([data_files[file_id]], col_idx)
        dr_tmp.read(delimiter='\t')
        data = dr_tmp.getData()
        data = parse_data(dp, data, col_name, target_col)
        
        generate_factor_map(output_raster_path, data, 6, col_name, RASTER_CONFIG)
        del data
        del dr_tmp
    