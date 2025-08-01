from codebleu.codebleu import calc_codebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF
from rouge_score import rouge_scorer
from Levenshtein import distance
from nltk.tokenize import word_tokenize
from crystalbleu import corpus_bleu

class CodeBleuCalculator:
    def __init__(self, lang='python'):
        self.lang = lang
        
    def calculate(self, references, hypothesis):
        
        if isinstance(references, str):
            references = [references]
            
        try:
           
            score = calc_codebleu([references], [hypothesis], self.lang, 
                                weights=(0.25, 0.25, 0.25, 0.25))
            return score
        except Exception as e:
            return 0.0


class BleuCalculator:
    def __init__(self):
        
        self.smoothing = SmoothingFunction().method1
        
    def calculate(self, references, hypothesis):

        if isinstance(references, str):
            references = [references]
            
        try:
           

            tokenized_refs = [word_tokenize(ref) for ref in references]
            tokenized_hyp = word_tokenize(hypothesis)
            
        
            score = sentence_bleu(tokenized_refs, tokenized_hyp,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=self.smoothing)
            return score
        except Exception as e:
            return 0.0


class RougeLCalculator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
    def calculate(self, references, hypothesis):
        if isinstance(references, str):
            references = [references]
            
        try:
            max_score = 0.0
            for ref in references:
                scores = self.scorer.score(ref, hypothesis)
                f1_score = scores['rougeL'].fmeasure
                max_score = max(max_score, f1_score)
            
            return max_score
            
        except Exception as e:
            return 0.0


class EditDistanceCalculator:
    def __init__(self):
        pass
        
    def calculate(self, references, hypothesis):

        if isinstance(references, str):
            references = [references]
        try:
            min_distance = float('inf')
            max_length = 0
           
            for ref in references:
                dist = distance(ref, hypothesis)
                min_distance = min(min_distance, dist)
                max_length = max(max_length, len(ref), len(hypothesis))
    
            if max_length == 0:
                return 0.0 
            
            normalized_score = min_distance / max_length
            return min(1.0, normalized_score)  
            
        except Exception as e:
            return 1.0  


class ChrFCalculator:
    def __init__(self, n_gram=6, beta=2):
        self.chrf = CHRF(char_order=n_gram, word_order=0, beta=beta)
        
    def calculate(self, references, hypothesis):

        if isinstance(references, str):
            references = [references]
            
        try:

            score = self.chrf.sentence_score(hypothesis, references)
            return score.score / 100.0  
        except Exception as e:
            return 0.0




class CrystalBLEUCalculator:
    def __init__(self):
        self.crystal_bleu = corpus_bleu
        
    def calculate(self, references, hypothesis):
        
        if isinstance(references, str):
            references = [references]
            
        try:
            refs_list = [[ref] for ref in references]  
            hyp_list = [hypothesis]  
            
            score = self.crystal_bleu(refs_list, hyp_list)
            return score
        except Exception as e:
            return 0.0
