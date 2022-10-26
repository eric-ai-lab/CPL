# coding: utf-8

import sys
dataDir = '/home/datasets/VQA_v2'
sys.path.insert(0, './lib/vqaTools')
from vqa import VQA
from lib.vqaEvaluation.vqaEval import VQAEval
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os

parser = ArgumentParser("VQAv2 Prompt")
parser.add_argument("--result", type=str, default="zero_shot")
parser.add_argument("--identifier", type=str, default="")
parser.add_argument("--question", action="store_true")
parser.add_argument("--prompt_type", type=str, default="")  # or ANSWER_PROMPT
parser.add_argument("--partition", type=int, default="8")
parser.add_argument("--models", type=int, nargs='+')

args = parser.parse_args()
print(f"Evaluate model {args.models}")

# set up file names and paths
versionType ='v2_' # this should be '' when using VQA v2.0 dataset
taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
dataSubType ='val2014'
annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    ='%s/InputQuestions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir      ='%s/InputImages/%s/%s/' %(dataDir, dataType, dataSubType)
resultType  =args.result
fileTypes   = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

collected_results = []
for j in args.models:
	print(f"Evaluate model at epoch {j}")
	result = []
	for i in range(args.partition):
		if args.prompt_type != "":
			with open(f'/home/data/vqa_results/results/vqa_dev_{args.prompt_type}_{i}_m{j}{args.identifier}.json') as tmp:
				result += json.load(tmp)
		else:
			with open(f'/home/data/vqa_results/results/vqa_dev_{i}_m{j}.json') as tmp:
				result += json.load(tmp)
	with open('/home/data/vqa_results/results/dev_results.json','w') as tgt:
		json.dump(result,tgt)
	print('Merged results have been writen at /home/data/vqa_results/results/dev_results.json')

	resFile = '/home/data/vqa_results/results/dev_results.json'

	[accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['../Results/%s%s_%s_%s_%s_%s.json'%(versionType, taskType, dataType, dataSubType, \
	resultType, fileType) for fileType in fileTypes]

	# create vqa object and vqaRes object
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)

	# create vqaEval object by taking vqa and vqaRes
	vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
	# evaluate results
	"""
	If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
	By default it uses all the question ids in annotation file
	"""
	vqaEval.evaluate()
	collected_results.append(vqaEval)
	print("\tOverall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))

	print("\tPer Answer Type Accuracy is the following:")
	for ansType in vqaEval.accuracy['perAnswerType']:
		print("\t%s (%.02f %%) : %.02f" % (
		ansType, vqaEval.accuracy['AnswerTypeStat'][ansType], vqaEval.accuracy['perAnswerType'][ansType]))
	print("\n")

for i, vqaEval in zip(args.models, collected_results):
	print(f"Acc of model at epoch {i}")
	# print accuracies
	print("\tOverall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))

	if args.question:
		print("\tPer Question Type Accuracy is the following:")
		for quesType in vqaEval.accuracy['perQuestionType']:
			print("\t%s (%.02f %%) : %.02f" %(quesType, vqaEval.accuracy['QuestionTypeStat'][quesType], vqaEval.accuracy['perQuestionType'][quesType]))
		print("\n")

	print("\tPer Answer Type Accuracy is the following:")
	for ansType in vqaEval.accuracy['perAnswerType']:
		print("\t%s (%.02f %%) : %.02f" %(ansType, vqaEval.accuracy['AnswerTypeStat'][ansType], vqaEval.accuracy['perAnswerType'][ansType]))
	print("\n")
