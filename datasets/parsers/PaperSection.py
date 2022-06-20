import re
import json
import stanza
import numpy as np
from .Paper import Paper

def clean_text(input):
    cleaned = input.strip()
    cleaned = re.sub("\n([0-9]*( *\n))+", "\n", cleaned)
    return cleaned

def findMajority(arr, size):
    m = {}
    for i in range(size):
        if arr[i] in m:
            m[arr[i]] += 1
        else:
            m[arr[i]] = 1
    count = 0
    for key in m:
        if m[key] > size / 2:
            count = 1
            print("Majority found :-",key)
            return 0
            break
    if(count == 0):
        print("No Majority element")
        return 1
    
def split_sentences(text, tokenizer=None):
    if tokenizer is not None:
        section_sentences = []
        for sent in tokenizer(text).sentences:
            sent_txt = []
            if len(sent.words) > 1:
                for word in sent.words:
                    sent_txt.append(word.text)
                sent_txt = ' '.join(sent_txt)
                section_sentences.append(sent_txt)
    else: #(?<!\d\.)
        section_sentences = re.split(r'(?<!\w\.\w.)(?<!\s\.)(?<!Eqn\.)(?<!Eqns\.)(?<!Sec\.)(?<!Tab\.)(?<!Fig\.)(?<!Figs\.)(?<!eq\.)(?<!Eqs\.)(?<!vs\.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<!\s[a-z]\.)(?<!\sal\.)(?:(?<=\.)|(?<=\!)|(?<=\?)|(?<=\w\:))\s', text)
        section_sentences = [s.replace('\n', ' ') for s in section_sentences if len(s.split()) > 1]

    return section_sentences

class PaperSection:
    def __init__(self, PAPER):
        self.ID = PAPER.ID
        self.TITLE = PAPER.TITLE
        self.ABSTRACT = PAPER.ABSTRACT
        self.SCIENCEPARSE = PAPER.SCIENCEPARSE
        self.SECTIONS = {
            'Abstract': None,
            'Introduction': None,
            'Related work': None,
            'Experiment': None,
            'Conclusion': None
        }
        self.SECTION_SENTENCES = {
            'Abstract': None,
            'Introduction': None,
            'Related work': None,
            'Experiment': None,
            'Conclusion': None
        }
        self.SECTION_INTENTS = {
            'Abstract': None,
            'Introduction': None,
            'Related work': None,
            'Method': None,
            'Experiment': None,
            'Conclusion': None
        }
        self.SCORE = {
            'RECOMMEND': None, 
            'SUBSTANCE': None, 
            'APPROPRIATE': None,
            'COMPARISON': None,
            'SOUNDNESS': None,
            'ORIGINALITY': None,
            'CLARITY': None, 
            'IMPACT': None,
        }
        self.aspect_keys = [
            'RECOMMENDATION',
            'SUBSTANCE', 
            'APPROPRIATENESS',
            'MEANINGFUL_COMPARISON',
            'SOUNDNESS_CORRECTNESS',
            'ORIGINALITY',
            'CLARITY', 
            'IMPACT',
        ]

    @staticmethod
    def from_paper(paper, tokenizer=None):
        paper_section = PaperSection(paper)

        sections = paper.SCIENCEPARSE.get_sections_dict()
        section_num = -1
        add_section = None
        for section_title, section_content in sections.items():
            add = True
            if section_num == section_title[0]:
                cleaned_text = clean_text(section_content)
                if paper_section.SECTIONS[add_section] is None:
                    paper_section.SECTIONS[add_section] = cleaned_text
                else:
                    paper_section.SECTIONS[add_section] += '\n' + cleaned_text
                add = False
            for section_key in paper_section.SECTIONS.keys():
                if section_key.lower() in section_title.lower():
                    if add:
                        cleaned_text = clean_text(section_content)
                        section_num = section_title[0]
                        add_section = section_key
                        if len(cleaned_text) > 0:
                            if paper_section.SECTIONS[add_section] is None:
                                paper_section.SECTIONS[add_section] = cleaned_text
                            else:
                                paper_section.SECTIONS[add_section] += '\n' + cleaned_text
                            add = False
                            
        if paper_section.ABSTRACT is not None:
            paper_section.SECTIONS['Abstract'] = paper_section.ABSTRACT
            paper_section.SECTION_SENTENCES['Abstract'] = split_sentences(paper_section.ABSTRACT)

        for section_key in paper_section.SECTIONS.keys():
            if paper_section.SECTIONS[section_key] is not None:
                paper_section.SECTION_SENTENCES[section_key] = split_sentences(paper_section.SECTIONS[section_key])

        if paper.REVIEWS:
            for aspect, key in zip(paper_section.SCORE.keys(), paper_section.aspect_keys):
                aspect_score = []
                for review in paper.REVIEWS:
                    if key in review.__dict__ and review.__dict__[key] is not None:
                        aspect_score.append(float(review.__dict__[key]))
                if aspect_score:
                    paper_section.SCORE[aspect] = {
                        'mean': np.mean(aspect_score),
                        'major': int(np.bincount(aspect_score).argmax())}
                    # if aspect == 'CLARITY':
                    #     j = findMajority(aspect_score, len(aspect_score))
                    #     print(aspect_score, )

        return paper_section

    def get_section(self, section_title):
        try:
            return self.SECTIONS[section_title]
        except KeyError:
            raise KeyError

    def get_section_sentences(self, section_title):
        try:
            return self.SECTION_SENTENCES[section_title]
        except KeyError:
            raise KeyError

    def set_section_sentences_intents(self, section_title, intents):
        self.SECTION_INTENTS[section_title] =  intents

    def set_method_from_file(self, method_file):
        self.SECTIONS['Method'] = None
        self.SECTION_SENTENCES['Method'] = None
        with open(method_file, 'r') as f:
            data = json.load(f)
        method_dict = next((item for item in data if item['ID'] == self.ID), None)
        if method_dict is not None:
            if method_dict['METHOD'] is not None:
                for section in method_dict['METHOD']:
                    if section['is_method'] == 1:
                        cleaned_text = clean_text(section['text'])
                        if len(cleaned_text) > 0:
                            if self.SECTIONS['Method'] is None:
                                self.SECTIONS['Method'] = cleaned_text
                            else:
                                self.SECTIONS['Method'] += '\n' + cleaned_text
        if self.SECTIONS['Method'] is not None:
            section_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\sal\.)(?<=\.|\!|\?|:)\s', self.SECTIONS['Method'])
            section_sentences = [s.replace('\n', ' ') for s in section_sentences if len(s.split()) > 1]
            self.SECTION_SENTENCES['Method'] = section_sentences