import os
import datetime
import json
import csv
import pandas as pd
import numpy as np
import bigfinite_api_client as api_client


#to get client --->
# client = api_client.api_helper.get_environment('curia') 
# 'curia' can be changed for another client, 
# make sure to have the correct password 
# in the .config/BigfiniteAPIClient/bigfinite_api_client.cfg file,
# also verify your VPN is ON

def ensure_dir_exists(directory):
    if os.path.exists(directory) == False:
        os.mkdir(directory)
        print("Creating Directory: ", directory)
    else:
        print("Directory Exists: ",  directory) 
        pass

def download_data_from_context_table (client, 
                                      phase_path:str, 
                                      phase_context_table:pd.DataFrame, 
                                      overwrite=False,
                                      limit=20000,
                                      entity_type_context_table_delimiter='__'):
    ''' Downloads data from platform entities using bigfinite_api_client.
        Uses a phase context table to download each entity (device) listed in the columns variable_x for each batch
        using the start timestamp and the end timestamp.

        Contex table of each phase has header:
        Batch , Start , End , Variable_1 , Variable_2 , ... , Variable_n
        and values:
        Batch_code , Start_timestamp , End_timestamp, Device_1 , Device_2 , ... , Device_n
        ------
        client: bigfinite_api_client for a particular environment
        phase_path: path for current phase context table
        phase_context_table: DataFrame with context information
        overwrite: overwrite file
        limit: maximum ammount of samples to download per batch and entity (between Start_timestamp and End_timestamp)
        entity_type_context_table_delimiter: this delimiter separates in the column names of the context table the
            entity type (devices, elements, etc.) from the rest of the column name, so the entity type is identified.

    '''
    ensure_dir_exists(phase_path)

    phase_context = phase_context_table.to_dict('records') #list of dicts for each record
    
    for row in phase_context:
        record = row
        batch = str(record.pop('Batch'))
        t_start = record.pop('Start')
        t_end = record.pop('End')
        
        if not pd.isna(t_start) and not pd.isna(t_end):
            for variable, entity  in record.items():
                if entity_type_context_table_delimiter in variable:
                    entity_type = variable.split(entity_type_context_table_delimiter)[0]
                else:
                    entity_type = 'devices' # it defaults to 'devices' for compatibility with old code
                if not pd.isna(entity):
                    output_path = f'{phase_path}/{batch}/{variable}'
                    output_file_name = f'{output_path}/{batch}_{entity}.csv' #arbitrary csv name

                    if not os.path.exists(output_path):
                        print(f'Creating {output_path}')
                        os.makedirs(output_path) #make directory tree
                        api_client.api_helper.get_entities_timeseries(client, [entity], [entity_type], 
                                                                        str(t_start), str(t_end),'csv',output_file_name,limit=limit)
                    elif (not os.path.exists(output_file_name)) or (os.path.exists(output_file_name) and os.path.getsize(output_file_name) <= 1):
                        print(f'File was empty. Downloading again {output_file_name}')                        
                        if os.path.exists(output_file_name):
                            os.remove(output_file_name)
                        api_client.api_helper.get_entities_timeseries(client, [entity], [entity_type], 
                                                                        str(t_start), str(t_end), 'csv', output_file_name,limit=limit)
                    elif os.path.exists(output_file_name) and os.path.getsize(output_file_name) > 0 and overwrite:
                        print(f'Overwriting {output_file_name}')                        
                        os.remove(output_file_name)
                        api_client.api_helper.get_entities_timeseries(client, [entity], [entity_type], 
                                                                        str(t_start), str(t_end), 'csv', output_file_name,limit=limit)
                    else:
                        print(f'Skipping existing and non-empty file {output_file_name}')


    return None             