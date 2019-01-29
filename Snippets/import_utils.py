import os
import re
import pandas as pd
from numpy import float32


# TODO does not work always
def import_files(directory: str, extension: str = 'csv', verbose: bool = False, files_excluded: list = None) -> pd.DataFrame:
    """ Read all files that have the extension from the folder and returns pd.DataFrame with idx
    :rtype: pd.DataFrame
    :return: dataframe with all files imported and idx as file names
    :param files_excluded: List of files that will be excluded from the import
    :param directory: the path to directory to extract the files
    :param extension: the type of file you wnat to include
    :param verbose: print on console files that are imported
    :type directory: path str
    """
    # extract all csv files
    fileList = list_files(directory, extension)
    files = list(fileList)

    # we remove 'idx_submission.csv' if present
    len_before = len(files)
    # we dont want that
    files = [file for file in files if file not in files_excluded]
    # files = [file for file in files if file != 'idx_submission.csv']
    len_after = len(files)
    if len_before - len_after == 0:
        pass
    elif len_before - len_after == 1:
        print('idx_submission.csv file has been removed from the list')
    else:
        raise IOError('More than one file has been removed')

    # extract file name only without extension to name each dataframe
    db_names = [re.search('(?<=)(.*)(?=.{})'.format(extension), file).group() for file in files] # group() gives string
    database = []
    # give to dataframes the filenames = keys of dictionary database
    # key: dataframe name value: corresponding csv.file
    for i, (name, file) in enumerate(zip(db_names, files)):
        print('Importing {}'.format(file))
        database.append(pd.read_csv(os.path.join(directory, file),
                                    skiprows=[0],
                                    header=None,
                                    index_col=0,
                                    names=[name],
                                    dtype={0: 'str', 1: float32}))
        if verbose:
            print(database[i].head())
            print(database[i].dtypes)
    database_rd: pd.DataFrame = pd.concat(database, axis=1, sort=True)
    return database_rd


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))
