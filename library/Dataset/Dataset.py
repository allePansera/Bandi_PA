import pandas as pd
import csv


class Dataset:
    """This class is used to import a specific dataset from local resource.
    The default file is inside source directory"""

    def __init__(self, path='source/dataset.csv'):
        """
        Constructor set the url as library attribute.
        :param url: url to use in order to download the dataset
        """

        self.path = path

    def dataset_format(self, save=False):
        """
        Method dataset_format() read csv file, prepare it and store is in a 'DataFrame' likely format.
        :param save: is True DataFrame downloaded is stored as a file.
        :return: DataFrame object built after read_csv method call
        """

        file = open(self.path, "r", encoding='iso-8859-1')
        csv_reader = list(csv.reader(file, delimiter=','))
        names = csv_reader.pop(0)
        content = csv_reader.copy()
        file.close()
        df = pd.DataFrame(data=[row for row in content], index=range(len(content)), columns=names)
        df = df.astype(str)
        if save:
            self.store_dataframe(df)
        return df

    def store_dataframe(self, df, path='./dataset/data.{}'):
        """
        Method store a dataframe inside a Path. DataFrame object is stored as CSV and XLSX.
        Path format is checked as df type.
        :param df: Dataframe instance to save as file.
        :param path: path where DataFrame is saved. Format must be omitted.
        :return: Nothing
        """
        if '.{}' not in path:
            raise Exception("Wrong 'path' parameter. Do not specify file type, use {} instead.")
        if not isinstance(df, pd.DataFrame):
            raise Exception("'df' parameter must be a Dataframe instance.")
        df.to_csv(path.format('csv'), index=False, encoding='iso-8859-1')
        # TO BIG 4 US: df.to_excel(path.format('xlsx'), merge_cells=False, index=True, encoding='iso-8859-1', engine='xlsxwriter')


