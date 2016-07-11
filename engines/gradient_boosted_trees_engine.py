# -*- coding:utf-8 -*-


from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

import os

class GradientBoostedTreesEngin:
    def __train_model(self):
        self.mode = GradientBoostedTrees.trainClassifier(self.trainingData, categoricalFeaturesInfo={}, numIterations=3)
    def __init__(self, sc, dataset_path):
        self.sc = sc
        self.model = None
        self.trainingData = None
        if os.path.isfile('/tmp/GBT.model'):
            self.model = GradientBoostedTreesModel.load(self.sc,'/tmp/GBT.model')
        if not self.model:
            self.__train_model()
            shutil.rmtree('/tmp/GBT.model',ignore_errors=True)
            self.model.save(self.sc,'/tmp/GBT.model')
        
