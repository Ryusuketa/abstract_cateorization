import gokart
import luigi
from data.dataset_parser import parse_dataset
from model.data.utils import calculate_transition_matrix, get_glove_vectors
from model.classifier.model_conf import SentenceCategoryLabels


class MakeSentenceData(gokart.TaskOnKart):
    pubmed_rct_path = luigi.Parameter(default='localfile/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt')

    def output(self):
        return self.make_target('data/pubmed_data.pkl')

    def run(self):
        df = parse_dataset(self.pubmed_rct_path)
        self.dump(df)


class MakeTransitionMatrix(gokart.TaskOnKart):
    def requires(self):
        return MakeSentenceData()

    def output(self):
        return self.make_target('data/transition_matrix.pkl')

    def run(self):
        df = self.load_data_frame()
        section2label = SentenceCategoryLabels().section2label
        transition_matrix = calculate_transition_matrix(df, section2label)
        self.dump(transition_matrix)


class PrepareWordEmbedding(gokart.TaskOnKart):
    embedding_txt = luigi.Parameter('localfile/glove.6B.200d.txt')

    def output(self):
        return self.make_target('data/embeddings.pkl')

    def run(self):
        embedding, token2id = get_glove_vectors(self.embedding_txt)
        dic_embedding = dict(embedding=embedding, token2id=token2id)
        self.dump(dic_embedding)
