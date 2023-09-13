import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from zipfile import ZipFile

import pandas as pd

from bigfinite_api_client.bigfinite import BigfiniteAPI

        
def download_data(client: BigfiniteAPI, params: Dict[str, Any], download_path: Path = None) -> str:
    """Function to download data from the platform using the data alignment API

    It calls the API to create a job, then polls the API every second until the job is finished and
    then downloads or returns error message depending on the output.

    Parameters
    ----------
    client : BigfiniteAPI
        A logged in client to the API
    params : Dict[str, Any]
        Parameters to pass to the API, see the Data Alignment DOM for more information 
    download_path : str, optional
        Where to download the result of the job. If none, data won't be downloaded. By default None

    Returns
    -------
    str
        The response from the API, which can be 'IN_PROGRESS', 'Downloaded' or an error message
    """
    create_response = client.create_data_alignment_job(params)
    print(f"Create Response: {create_response}")

    get_response = "IN_PROGRESS"
    repeat_count = 0
    while get_response == "IN_PROGRESS":
        get_response = client.get_data_alignment_job(create_response, None)
        print(f"Get Response: {get_response} | {repeat_count}")
        repeat_count += 1
        time.sleep(1)

    if get_response == 'Downloaded' and download_path:
        with NamedTemporaryFile() as tmp_file:
            client.get_data_alignment_job(create_response, tmp_file.name)

            with ZipFile(tmp_file.name, 'r') as zip_ref:
                download_path.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(download_path)

    elif get_response == "FAILED":
        get_response = client.get_data_alignment_job(create_response, None, get_error_message=True)

    return get_response


def concatenate_data(
    data_path: Path,
    final_path: Path,
    remove_data_path: bool = True,
    timestamps_to_datetime: bool = True,
    read_csv_kwargs: Dict[str, Any] = {'sep': '\t', 'header': 0, 'index_col': [0, 1]},
    to_csv_kwargs: Dict[str, Any]= {'sep': ';'},
) -> str:
    """String together the data downloaded from a data alignment job

    Parameters
    ----------
    data_path : Path
        Where the output of the data alignment job is stored
    final_path : Path
        Where to store the final concatenated data
    remove_data_path : bool, optional
        If the original data should be removed, by default True
    timestamps_to_datetime : bool, optional
        If the timestamps should be converted to datetime and the slice index dropped, by default
        True
    read_csv_kwargs : Dict[str, Any], optional
        Arguments to pass to the pd.read_csv function, by default {'sep': '\t', 'header': 0,
        'index_col': [0, 1]}
    to_csv_kwargs : Dict[str, Any], optional
        Arguments to pass to the pd.to_csv function, by default {'sep': ';'}

    Returns
    -------
    str
        A message indicating the end of the function
    """
    final_path.unlink(missing_ok=True)

    with open(next(data_path.iterdir())) as f:
        header = f.readline()

    columns_names = header.strip().split("\t")
    df = pd.DataFrame(columns=columns_names)
    df.set_index(columns_names[0:2], inplace=True)

    for data_file in data_path.iterdir():
        aux_df = pd.read_csv(data_file, **read_csv_kwargs)
        df = pd.concat([df, aux_df])

    df.sort_index(inplace=True)
    if timestamps_to_datetime:
        df.index = df.index.get_level_values(1).map(lambda x: datetime.fromtimestamp(x / 1000.0))

    df.to_csv(final_path, **to_csv_kwargs)

    if remove_data_path:
        for data_file in data_path.iterdir():
            data_file.unlink()
        data_path.rmdir()
    
    return "concatenated"