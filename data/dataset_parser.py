import pandas as pd
import itertools
import re


def parse_dataset(path: str):
    with open(path, 'r') as f:
        texts_org = f.read()

    texts = re.split('###[0-9]+\n', texts_org)
    paper_id = re.findall('###[0-9]+\n', texts_org)
    li_texts = []
    for id, text in zip(paper_id, texts):
        sentenses = text.split('\n')
        sentenses = [sentense for sentense in sentenses if len(sentense) > 3]
        sentenses = [[id.replace('#', '').replace('\n', '')] + sentense.split('\t') + [i] for i, sentense in enumerate(sentenses)]
        li_texts.append(sentenses)

    texts_line = list(itertools.chain.from_iterable(li_texts))

    text_id = [line[0] for line in texts_line]
    section_names = [line[1].lower() for line in texts_line]
    section_sentense = [line[2].lower() for line in texts_line]
    section_sent_position = [line[3] for line in texts_line]
    documents = pd.DataFrame(dict(paper_id=text_id, sentense=section_sentense, section=section_names, position=section_sent_position))

    return documents


if __name__ == '__main__':
    path = '/Users/ryusuke.tanaka/project/paper_constructor/sandbox/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/train.txt'

    data = parse_dataset(path)
