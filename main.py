import gokart
import luigi
import numpy as np
from model.classifier.train import SentenceClassifierModelTrain
from model.classifier.model_conf import SentenceClassifierModelParameters
from model.classifier.validation import ModelValidation

if __name__ == '__main__':
    np.random.seed(1)
    luigi.configuration.core.add_config_path('./conf/model.ini')
    luigi.build([ModelValidation()], local_scheduler=True)
