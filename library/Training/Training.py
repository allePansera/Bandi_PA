from library.Dataset.Dataset import Dataset
from library.Dataset.Normalization import Normalization
from library.Training.Classifier import Classifier
from library.Plot.ConfusionMatrix import plot as conf_matrix_plot
from library.Plot.FeatureImportance import plot as feature_importance
import time, logging
import pandas as pd


class Training:
    """Training class is used to control che training of the model
    """

    def __init__(self, method="RF", sampler="OS_SVM", logger='log/training.log'):
        """

        :param method: Classifier algorithm
        :param sampler: over-samplig / under-sampling used technique
        :param logger: path where to store log information
        """
        self.logger = logging.getLogger("training-logger")
        self.method = method
        self.sampler = sampler
        logging.basicConfig(filename=logger, filemode="w", level=logging.INFO, format="%(asctime)s; %(levelname)s: %(message)s")

    def train(self, read_existing=True, path_existing='dataset/data.csv',
              read_normalized=False, normalized_path="dataset/data_normalized.csv"):
        """
        Method is used to train the model with given constructor parameter.
        :param read_existing: if True then the df is built on existing file
        :param path_existing: path where existing content is stored to build DataFrame instance
        :param read_normalized: if True df is retrived directly from a file
        :param normalized_path: path where is stored normalized dataset
        :return: nothing
        """
        start = time.time()
        self.logger.info("Running DataFrame re-construction")
        d = Dataset()
        # Check if data need to be restored or if ain't necessary
        if read_existing and not read_normalized:
            df = pd.read_csv(path_existing, dtype=object)
            end = time.time()
            self.logger.info(f"DataFrame read in {round(end - start, 2)}sec")

        elif not read_existing and not read_normalized:
            df = d.dataset_format(save=True)
            d.store_dataframe(df)
            end = time.time()
            self.logger.info(f"DataFrame stored in {round(end - start, 2)}sec")

        elif read_normalized and read_existing:
            raise Exception('"read_existing" and "read_normalized" cannot be both True')


        if not read_normalized:
            start = time.time()
            self.logger.info("Running DataFrame normalization")
            n = Normalization(df)
            df = n.execute(save=True)
            end = time.time()
            self.logger.info(f"DataFrame normalized & stored in {round(end - start, 2)}sec")
        else:
            df = pd.read_csv(normalized_path, skiprows=[0])

        self.logger.info(f"Classifier production - {self.method}...")
        cl = Classifier(df, self.method, self.sampler)
        start = time.time()
        cl.train()
        cm, f1, good_borrow_precision, bad_borrow_precision, fpr, precision, threshold, model = cl.test()
        end = time.time()
        self.logger.info(f"Classifier produced in {round(end - start, 2)}sec")

        conf_matrix_plot(cm)
        # roc_curve(threshold, model)
        if hasattr(cl.get_classifier(), "feature_importances_"):
            feature_importance(cl.get_classifier().feature_importances_)

        # USELESS: correlation_plt(df)
        cl.save_classifier()

        self.logger.info(f"F1 score: {f1}")
        self.logger.info(f"Good borrower prediction: {round(good_borrow_precision, 2)}%")
        self.logger.info(f"Bad borrower prediction: {round(bad_borrow_precision, 2)}%")
        self.logger.info(f"FDR: {round(fpr, 2) * 100}%")
        self.logger.info(f"Precision: {round(precision, 2) * 100}%")

        self.logger.info(f"Classifier stored...")

    def test(self):
        pass