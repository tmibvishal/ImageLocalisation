import os

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
