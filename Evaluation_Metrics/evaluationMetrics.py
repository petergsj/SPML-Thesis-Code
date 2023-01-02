import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score


def getBLEUScore(pred_answer, answer):
  max_bleu_score = 0

  candidate = nltk.word_tokenize(answer.lower().replace(".", " ")) #lowercasing and removing fullstop
  if type(pred_answer) != list:
    pred_answer = [pred_answer]
  assert type(pred_answer) == list, pred_answer+' of type ->'+type(pred_answer)

  for ans in pred_answer:
    reference = [nltk.word_tokenize(ans.lower().replace(".", " "))] #lowercasing and removing fullstop

    smooth_fn = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth_fn, weights=(1, 0, 0, 0))

    if bleu_score > max_bleu_score:
      max_bleu_score = bleu_score
      
  return max_bleu_score

def getMeteorScore(pred_answer, answer):
  max_meteor_score = 0
  candidate = nltk.word_tokenize(answer.lower().replace(".", " ")) #lowercasing and removing fullstop
  if type(pred_answer) != list:
    pred_answer = [pred_answer]
  assert type(pred_answer) == list, pred_answer+' of type ->'+type(pred_answer)
  for ans in pred_answer:
    reference = nltk.word_tokenize(ans.lower().replace(".", " ")) #lowercasing and removing fullstop
    meteor_score = single_meteor_score(reference, candidate)
    if meteor_score > max_meteor_score:
      max_meteor_score = meteor_score
  return max_meteor_score

def isExactMatch(pred_answer, answer):
  if type(pred_answer) != list:
    pred_answer = [pred_answer]
  assert type(pred_answer) == list, pred_answer+' of type ->'+type(pred_answer)
  for ans in pred_answer:
    if answer.lower() == ans.replace(".", "").lower():
        return True
  return False

def Score_Similarity_Evaluator(pred_answer, answer):
  bleuScore = getBLEUScore(pred_answer, answer)
  meteorScore = getMeteorScore(pred_answer, answer)
  isMatch = isExactMatch(pred_answer, answer)

  return {'bleuScore': bleuScore,
          'meteorScore': meteorScore, 
          'isMatch': isMatch}

