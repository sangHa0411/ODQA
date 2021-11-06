
import re

class Preprocessor :
    def __init__(self,) :
        self.base_sub = re.compile(r'\\|\n[*#]*')
        self.unk_sub = re.compile('[\u3000-\u303f\ud800—\udbff\ue000—\uf8ff]')
        self.outrange_sub = re.compile('[\uffff-\U000e007f]')

    def preprocess4train(self, dataset) :
        assert isinstance(dataset, dict)
        context = dataset['context']
        question = dataset['question']
        answer = dataset['answers']

        answer_start, answer_txt = answer['answer_start'][0], answer['text'][0]

        context_prev = context[:answer_start]
        context_next = context[answer_start+len(answer_txt):]

        context_prev = self.doc_preprocess(context_prev)
        context_next = self.doc_preprocess(context_next)
        answer_txt = self.doc_preprocess(answer_txt)

        answer_start = len(context_prev)

        dataset['context'] = context_prev + answer_txt + context_next
        dataset['answers'] = {'answer_start' : [answer_start], 'text' : [answer_txt]}
        dataset['question'] = self.doc_preprocess(question)
        return dataset

    def preprocess4test(self, dataset) :
        assert isinstance(dataset, dict)
        question = dataset['question']
        dataset['question'] = self.doc_preprocess(question)
        return dataset

    def doc_preprocess(self, txt) :
        txt = self.base_sub.sub(txt, ' ')
        txt = self.unk_sub.sub(txt, ' ')
        txt = self.outrange_sub.sub(txt, ' ')
        txt = re.sub('\s+', ' ', txt)
        return txt

