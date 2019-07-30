import gokart
import luigi
import numpy as np
from model.classifier.train import SentenceClassifierModelTrain
from model.classifier.train import confirm
from model.classifier.model_conf import SentenceClassifierModelParameters

if __name__ == '__main__':
    np.random.seed(1)
    luigi.configuration.core.add_config_path('./conf/model.ini')
    luigi.build([confirm(rerun=True)], local_scheduler=True)