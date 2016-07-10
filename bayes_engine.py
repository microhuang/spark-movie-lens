from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

import os
import shutil
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesEngine:

    def __train_model(self):
        self.model = NaiveBayes.train(self.ratings_RDD, 1.0)
        logger.info("Bayes model built!")

    def __init__(self, sc, dataset_path):
        logger.info("Loading Ratings data...")
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

        if os.path.isfile('/tmp/Bayes.model'):
            self.model = NaiveBayesModel.load(sc, '/tmp/Bayes.model')
        if not self.model:
            self.__train_model()
            shutil.rmtree('/tmp/Bayes.model',ignore_errors=True)
            self.model.save(self.sc,'/tmp/Bayes.model')

