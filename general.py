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
