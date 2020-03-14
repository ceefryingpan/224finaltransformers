from transformers import AlbertTokenizer, AlbertForQuestionAnswering, AlbertForSequenceClassification
import torch
import nltk
import json
import csv
import os

nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

home = os.path.expanduser("~")
with open(os.path.expanduser('~/224nfinalproject/data/dev-v2.0.json')) as reader:
    data = json.load(reader)["data"]
with open(os.path.expanduser('~/224finaltransformers/examples/output/albertforqa/nbest_predictions_.json')) as verifier:
    qamodel = json.load(verifier)
verifiertokenizer = AlbertTokenizer.from_pretrained(os.path.expanduser('~/224finaltransformers/examples/output/albertforsqc'))
verifiermodel = AlbertForSequenceClassification.from_pretrained(os.path.expanduser('~/224finaltransformers/examples/output/albertforsqc'))

def answerandprob(qamodelanswers):
    bestanswer = qamodelanswers[0]
    noansprob = 0.0

    if bestanswer["text"] == "":
        noansprob = bestanswer["probability"]
        bestanswer = qamodelanswers[1]
    else:
        for answer in qamodelanswers:
            if answer["text"] == "":
                noansprob = answer["probability"]

    return bestanswer["text"], noansprob

def snipcontext(context, answer):
    index = context.find(answer)
    context_list = sent_detector.tokenize(context.strip())
    chars_seen = 0
    context_sent = ''
    first = True
    first_sentence = True
    for sent in context_list:
        chars_seen += len(sent) + 1 # add one for the space
        if first:
            chars_seen -= 1
            first = False
        if first_sentence and index < chars_seen:
            context_sent += sent
            first_sentence = False
            if index + len(answer) <= chars_seen:
                break
        elif not first_sentence:
            context_sent += ' ' + sent
            if index + len(answer) <= chars_seen:
                break
    return context_sent

fields = ['Id', 'Predicted']
filename = "dev_outputs.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    count = 1
    for topic in data:
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                id = qa["id"]
                question = qa["question"]

                qamodelanswers = qamodel[id]
                answer, noansprob_para = answerandprob(qamodelanswers)

                context_sent = snipcontext(context, answer)

                classes = ["no answer", "has answer"]

                sequence_0 = question # question
                sequence_1 = context_sent # snipped context

                has_answer = verifiertokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")
                has_answer_classification_logits = verifiermodel(**has_answer)[0]
                has_answer_results = torch.softmax(has_answer_classification_logits, dim=1).tolist()[0]

                noansprob_sent = has_answer_results[0]

                if (noansprob_para + noansprob_sent) < 1.25:
                    csvwriter.writerow([id, answer])
                else:
                    csvwriter.writerow([id, ""])

                print("Done with example number {}".format(count))
                count += 1
