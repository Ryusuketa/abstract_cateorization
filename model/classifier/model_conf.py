import luigi


class SentenceClassifierModelParameters(luigi.Config):
    encoded_features = luigi.IntParameter()
    attention_features = luigi.IntParameter()
    attention_hop = luigi.IntParameter()
    dropout_rate = luigi.FloatParameter()


class SentenceClassifierTrainParameters(luigi.Config):
    epoch = luigi.IntParameter()
    minibatch_size = luigi.IntParameter()
    lr = luigi.FloatParameter()
    optimizer = luigi.Parameter()


class SentenceCategoryLabels(luigi.Config):
    section2label = luigi.DictParameter(default=dict(background=0,
                                                     objective=0,
                                                     methods=1,
                                                     results=2,
                                                     conclusions=3))
