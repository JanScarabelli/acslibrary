import os

def upload(client, path: str, model_dic_list: list, model_cards: bool = False) -> None:
    """
    Iterates through all the files in a specified directory. For each ONNX model file found,
    creates metadata using the information inside the dictionaries passed through the list `model_dic_list`,
    and uploads the file using the provided client. The fields 'model id' and 'description' are mandatory,
    while other fields are optional, and default values will be used if not specified.

    Parameters
    ----------
    client : object
        The client used to interact with the server.
    path : str
        The directory where the ONNX models are stored.
    model_dic_list : list
        A list of dictionaries where for each model they have as keys all any of the different attributs that we can fill when uploading a model. The model id and 
        description are compulsary.
    model_cards: bool
        A boolean value that indicates whether we are going to upload the model in an environment where the model cards version is deployed or not.
    Returns
    -------
    None

    Notes
    -----
    - The directory path is assumed to be valid and not encoded.
    - For each file in the directory:
        - The file name is decoded.
        - If the file is an ONNX model and its name is in models_desc:
            - The name of the model and its is obtained.
            - The full path to the model is generated.
            - A model entity is created using the all the fields that we set into the model dict input.
            - The model metadata is created on the server.
            - A presigned URL for the model file is obtained where the model will be uploaded.
            - The model file is uploaded to the obtained URL.
    """

    directory = path

    model_ids = {model["id"] for model in model_dic_list}

    # Create a dictionary mapping model name to model entity
    model_entities = {model['id']: create_model_entity(model, model_cards) for model in model_dic_list}

    match_model_id = False
    # Iterate over every file in the directory
    for file in os.listdir(directory):
        # Decode the file name
        filename = os.fsdecode(file)
        
        # If the file is an ONNX model and its name is in model_ids
        if filename.endswith(".onnx") and os.path.splitext(filename)[0] in model_ids:
            match_model_id = True
            # Get the name of the model
            name = os.path.splitext(filename)[0]

            # Generate the full path to the model
            file_path = os.path.join(directory, filename)

            # Get the corresponding model entity
            model_entity = model_entities.get(name)

            if model_entity:
                # Create the model metadata on the server
                client.create_model_meta(model_entity)  # comment in case you already created the entity but not upload any model

                # Get a presigned URL for the model file
                client.get_presigned_url(name)

                # Upload the model file to the obtained URL
                client.upload_to_url(file_path, name)

    if match_model_id == False:
        print("None of the .onnx model files in the provided path match any of the IDs.")

def create_model_entity(model_dic: dict, model_cards: bool) -> dict:
    """
    Create a model entity dictionary based on the provided model dictionary.

    Parameters
    ----------
    model_dic : dict
        A dictionary containing the model information.

    model_cards: bool
        A boolean value that indicates whether we are going to upload the model in an environment where the model cards version is deployed or not.

    Returns
    -------
    dict
        The model entity dictionary.

    Notes
    -----
    - The 'id' and 'description' fields are extracted from the model_dic dictionary.
    - If 'id' or 'description' is missing, an error message is printed.
    - The 'model_entity' dictionary is initialized with default values and the extracted 'id' and 'description'.
    - The remaining key-value pairs from model_dic are updated in 'model_entity'.
    - If a key in model_dic is not a valid key in 'model_entity', an error message is printed.
    """
    name = model_dic['id'] #id field is mandatory
    description = model_dic['description'] #description field is mandatory

    if model_cards:
        model_entity = {
            'id': name,
            'code': f'{name} CODE',
            'uom': {
                'description': 'Model output in JSON format',
                'magnitude': 'Inference',
                'name': 'Model output',
                'uom': 'Output unit'
            },
            'description': description,
            'active': True,
            'customTags': [],
            'isaTags': [],
            'geolocation': {
                'lat': 0,
                'lng': 0
            },
            'intendedUse': "Predict",
            'limitations': "Model degradation",
            'trainSet': {"performanceMetrics": []},
            'testSet': {"performanceMetrics": []},
        }

        for key in model_dic.keys():
            if key not in model_entity:
                print(f"Error: Dictionary key '{key}' is not valid. Use the same naming convention as in [API] Create an AI Model")

        model_entity.update((key, value) for key, value in model_dic.items() if key in model_entity)
    
    else:
        model_entity = {
            'id': name,
            'code': f'{name} CODE',
            'uom': {
                'description': 'Model output in JSON format',
                'magnitude': 'Inference',
                'name': 'Model output',
                'uom': 'Output unit'
            },
            'description': description,
            'active': True,
            'customTags': [],
            'isaTags': [],
            'geolocation': {
                'lat': 0,
                'lng': 0
            },
        }

        for key in model_dic.keys():
            if key not in model_entity:
                print(f"Error: Dictionary key '{key}' is not valid. Use the same naming convention as in [API] Create an AI Model")

        model_entity.update((key, value) for key, value in model_dic.items() if key in model_entity)

    return model_entity

