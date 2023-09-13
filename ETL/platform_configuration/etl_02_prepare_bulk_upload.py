# -*- coding: utf-8 -*-
"""
Created on Wed July 27 19:56:28 2022

@author: WilliamNadolski

NAME:       etl_02_prepare_bulk_upload.py
PURPOSE:    convert staged data into format required to support bulk data upload to platform
INPUTS:     staged input data to be uploaded
            entity configuration mapping file (df_mapping)
OUTPUTS:    one csv file per entity-chunk combination routed to outDir
DEPENDENCY: existence of configuration mapping file (entity_config_xlsx)
PARAMS:     outDir, input_file_data, entity_config_xlsx, max_rows_per_file
EXCEPTIONS: no exceptions assessed
LOGIC:      Script will load input file (df_data) and configuration mapping file (df_mapping)
            Then map input column names to entity tag values
            Chunk up file as appropriate and identify min and max dttm for each chunk
            Finally output one file per entity + chunk with standard naming convention
            
"""

### IMPORT PACKAGES ###

import os
import sys
import openpyxl
import numpy as np
import pandas as pd

###  DEFINE PARAMETERS ###

#define input/output locations
outDir = r"D:\data\geh\beDF\dbovd\output"
input_file_data = r"D:\data\geh\beDF\dbovd\input\dbovd_wide.xlsx"
entity_config_xlsx = r"D:\data\geh\config\GEH_DBOVD_TIMESERIES_ENTITIES.xlsx"

#define sheet details
add_z_to_dttm = True #add 'Z' suffix to make ISO8601 compliant
max_rows_per_file = 100000 #chunk large files
file_mapping_sheet = 'devices'
input_unique_dttm = 'DTTM_UNIQUE'
mapping_tag_col = '__Tag__'
mapping_id_col = 'id'
col_suffix = '|3'

### COMPILE FUNCTIONS ###

# Reading mapping tag vs id needed for the csv header supported by Aizon id|type
def get_mapping(path_file:str, sheet_name:str):
    return pd.read_excel(path_file, sheet_name=sheet_name, engine='openpyxl')

# the primary timestamp column will be based upon input_unique_dttm for all the csv files generated
def get_data(path_file:str, timestamp_column:str):
    return pd.read_excel(path_file).rename(columns={timestamp_column:"timestamp"})

#Rename columns on df_data accoring to the id from the mapping 
def rename_cols(df_mapping, df_data, col_suffix=col_suffix):
    dict_rename = dict(zip(df_mapping[mapping_tag_col], (df_mapping[mapping_id_col] + col_suffix)))
    df_data = df_data.rename(columns=dict_rename)
    return df_data

#chunk out records, construct file name, and output csv file
def write_to_csvs(data, file_name, max_rows=10000000000, export_index=False):
    output_path = os.path.join(outDir, '{file_name}_{mindttm}_to_{maxdttm}.csv')
    groups = data.groupby(np.arange(len(data.index))//max_rows)
    #loop over groupings and identify min/max dttm for each group
    for (frameno, frame) in groups:
        #sanitize colons, periods, and Z suffix from dttm for filenaming purposes
        min_dttm = frame['timestamp'].min().replace(':','-').replace('.','-').replace('Z','') 
        max_dttm = frame['timestamp'].max().replace(':','-').replace('.','-').replace('Z','') 
        targetname = output_path.format(file_name=file_name, mindttm=(min_dttm), maxdttm=(max_dttm))
        frame.to_csv(targetname, index=export_index)
        print("SUCCESS: Output File:", targetname)

# Generate a file per column excepting the timestamp (last column)
def create_files(df_data, add_z_to_dttm=False):
    #optionally add Z suffix to dttm to make ISO8601 compliant
    if add_z_to_dttm == True:
        df_data['timestamp'] = df_data['timestamp'].str.strip() + 'Z'
    elif add_z_to_dttm == False:
        pass
    #loop over each column and output one csv file per device + chunk
    for col in df_data.columns[:-1]: #exclude timestamp col which is included in all files
        file_name = col.replace(col_suffix, '').strip() #unable to use pipe char in filenames on windows
        write_to_csvs(df_data[['timestamp', col]], file_name, max_rows_per_file)

### EXECUTE FUNCTIONS ###

# Execute Functions: Load Dataframes, Rename Cols, Output Files
df_mapping = get_mapping(entity_config_xlsx, file_mapping_sheet)
df_data = get_data(input_file_data, input_unique_dttm)
df_data = rename_cols(df_mapping, df_data)
create_files(df_data, add_z_to_dttm=True)

