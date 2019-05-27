import os
import pickle

def ensure_path(path: str):
    """Takes the path as a string and ensures the path exists. 
    If not then the path is created

    Parameters
    ----------
    path : The path as a string

    Returns
    -------
    Bool : True if the path exists/ is created
    False if any error occured
    """

    # handling case like /folder1/folder2
    if (str[0] == "." and str[1] == "/"):
        str = str[1:]

    # handling case like ./folder1/folder2
    if(str[0] == "." and str[1] == "/"):
        str = str[2:]

    # handling case like folder1/folder2/
    n = len(str)
    if(str[n-1] == '/'):
        str = str[:-1]

    folders = path.split('/')
    current_path = ''
    for fld in folders:
        if current_path == '':
            current_path = fld
        else:
            current_path = current_path + '/'+ fld
        if not os.path.isdir(current_path):
            try:
                os.mkdir(current_path)
            except:
                return False
    return True


def load_from_memory(file_name: str, folder: str = None):
    """
    Load a python object saved as .pkl from the memory

    :param file_name: name of the file
    :param folder: name of the folder, folder = None means current folder
    :return: pyobject or False if fails to load
    """
    try:
        with open(folder + "/" + file_name, 'rb') if folder is not None else open(file_name, 'rb') as input_rb:
            pyobject = pickle.load(input_rb)
            return pyobject
    except Exception as exception:
        return False, exception


def save_to_memory(pyobject, file_name: str, folder: str = "."):
    """
    Save a pyobject to the memory

    :param pyobject: python object
    :param file_name: file_name that pyobject will be saved with
    :param folder: name of the folder where to save, folder = None means current folder
    :return: True if file is loader or False otherwise
    """

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    try:
        with open(folder + "/" + file_name, 'wb') if folder is not None else open(file_name, 'wb') as output:
            pickle.dump(pyobject, output, pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        raise e
