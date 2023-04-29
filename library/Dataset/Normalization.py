"""
Normalization rules
- Feat. 1: dt_upd (string - dd/mm/yyyy)
- Feat. 2: oggetto gara (string - valutare estrazioni singole parole)
- Feat. 3: settore (da classificare, stringa)
- Feat. 4: modalità realizzazione (da classificare? stringa)
- Feat. 5: importo (verificare se va convertito da stringa a float)
- Feat. 6: num_tot_lotti (verificare se va convertito da stringa a float)
- Feat. 7: rup (da classificare? stringa)
- Feat. 8: codice_fiscale_stazione_appaltante (da classificare - stringa)
- Feat. 9: codice_istat_stazione_appaltante (da classificare - stringa)
- Feat. 10: somma_urgenza (da classificare - stringa)
- Feat. 11: tipo_appalto (da classificare - stringa)
- Feat. 12: tipo_procedura (da classificare - stringa)
- Feat. 13: criterio_aggiudicazione (da classificare - stringa)
- Feat. 14: imp_lotto_netto_sicurezza (verificare se va convertito da stringa a float)
- Feat. 15: imp_sicurezza (verificare se va convertito da stringa a float)
- Feat. 16: imp_lotto (verificare se va convertito da stringa a float)
- Feat. 17: categoria_prevalente * DA CAPIRE COSA CAZZO SIA *
- Feat. 18: classifica (da classificare - stringa; sono numeri romani) * IN TEORICA NON MI SERVE A NIENTE SICCOME NON E' NOTO A PRESCIDNERE *
- Feat. 19: tipo_atto_o_documento (da classificare - stringa)
- Feat. 20: aggiudicatario * VEDERE SE USARLO PER TARGET *
- Feat. 21: cf_aggiudicatario * VEDERE SE USARLO PER TARGET *
"""
import numpy as np

from library.Dataset.Dataset import Dataset
import pandas as pd
import json


class Normalization:
    """
    Normalization class executes data transformation based on above rules
    """

    def __init__(self, df: pd.DataFrame, config_path='library/Dataset/rules/config.json', replacing_path='library/Dataset/rules/replace_config.json'):
        """
        :param df: DataFrame to normalize
        :param config_path: path where rules are stored
        :param replacing_path: path where replacing rules are stored
        """
        self.df = df
        self.config_path = config_path
        self.replacing_path = replacing_path

    def execute(self, save=False):
        """
        Execute DataFrame transformation
        :param save: var. used to save or not new normalized DataFrame
        :return: updated DataFrame
        """
        # replacing 4 categorical features
        self.replacing()
        # target var analysis and adapting
        self.replacing_target()
        # discard all unnecessary features
        self.feature_selection()
        # remove row with Nan attributes
        self.remove_nan_row()
        # store, if required, normalized DataFrame
        if save:
            d = Dataset()
            d.store_dataframe(self.df, path='dataset/data_normalized.{}')

        return self.df

    def replacing(self):
        """
        replacing() method is used to apply replace for categorical features
        :return: nothing
        """
        file = open(self.config_path, "r")
        rules = json.load(file)
        file.close()
        file_out = open(self.replacing_path, "w")

        replacing = {"rules": []}
        for col in rules['cols']:
            if col["replacing"]:
                replacement = self.__get_key_categories(col["key"])
                replacing["rules"].append(replacement)

        json.dump(replacing, file_out, indent=4, sort_keys=True)
        file_out.close()

        # apply replacing to the DataFrame
        for rule in replacing["rules"]:
            if "target" not in rule:
                key = rule["key"]
                to_replace = {}
                for feature_value_dict in rule["values"]:
                    to_replace[feature_value_dict["original"]]=feature_value_dict["replace"]
                self.df[key] = self.df[key].replace(to_replace=to_replace)
                # set 0 for null values
                self.df[key] = self.df[key].fillna(0)
                self.df[key].astype(dtype='int64')

    def replacing_target(self, target_col='cf_aggiudicatario'):
        """
        replacing_target() method is used to replace existing values with one and empty one with 0
        :return: nothing
        """
        # set 0 for null values
        self.df[target_col] = self.df[target_col].apply(lambda x: 1 if not pd.isnull(x) else 0)
        self.df[target_col].astype(dtype='int64')

    def feature_selection(self):
        """
        This method apply a mask to DataFrame instance in order to keep only feature inside config.json file
        :return: nothing
        """
        file = open(self.config_path, "r")
        rules = json.load(file)["cols"]
        file.close()
        accepted_keys = [rule["key"] for rule in rules]
        self.df = self.df[self.df.columns.intersection(accepted_keys)]

    def remove_nan_row(self):
        """
        This method remove rows whose contain NaN attribute
        :return: nothing
        """
        self.df = self.df.dropna()

    def __get_key_categories(self, key):
        """
        Used to analyze class DataFrame feature values
        :param key: Dataframe key to search for classes
        :return: dictionary {'key':'feature_key', 'values':
            [
                {
                    'original':'',
                    'replace':0,
                    'qty': 0
                },
                ...
            ]
        }
        """
        output = {"key": key, "values": []}
        res = dict(self.df[key].value_counts())
        for index, inner_key in enumerate(res):
            temp = {"original": inner_key, "replace": int(index)+1, "qty": int(res[inner_key])}
            output["values"].append(temp)
        return output

