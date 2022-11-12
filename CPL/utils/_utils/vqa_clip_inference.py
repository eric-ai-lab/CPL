import sys
from lib.vqaTools.vqa import VQA
import json
import stanza

import torch
import clip
import os
import inflect
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

parser = ArgumentParser("VQAv2 Prompt")
parser.add_argument("--yesno", action="store_true")
parser.add_argument("--number", action="store_true")
parser.add_argument("--other", default="True", action="store_true")
parser.add_argument("--auto_template", action="store_true")
parser.add_argument("--clip", type=str, default="ViT-B/32")
parser.add_argument("--result", type=str, default="zero_shot")
parser.add_argument("--identifier", type=str, default="")
parser.add_argument("--partition", type=int, default="64")
parser.add_argument("--portion", type=int, default="0")
parser.add_argument("--mount", type=str, default="/mnt")
parser.add_argument("--prompt_type", type=str, default="LM_PROMPT")  # or ANSWER_PROMPT
parser.add_argument("--mask_token", type=str, default="<extra_id_0>") # BERT: [MASK] T5: <extra_id_0> BART: <mask>

parser.add_argument("--checkpoint", type=str, default="DATA/MODELS/model_epoch_")
parser.add_argument("--mid", type=str, default='0')
parser.add_argument("--few-shot", action="store_true")
parser.add_argument("--finetuning", action="store_true")
parser.add_argument("--oracle", action="store_true")

args = parser.parse_args()
print(f'Evaluate on yes/no: {str(args.yesno)}')
print(f'Evaluate on number: {str(args.number)}')
print(f'Evaluate on other: {str(args.other)}')
print(f'CLIP model: {str(args.clip)}')
print(f'Result type: {str(args.result)}')

nums = [str(i) for i in range(101)]
years = [str(i) for i in range(1950,2015)]

# stanza.download('en')
# nlp_pipeline = stanza.Pipeline('en')


dataDir		= f'/data1/xh/vqa'
versionType = 'v2_' # this should be '' when using VQA v2.0 dataset
taskType    = 'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
annFile     = '%s/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
quesFile    = '%s/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
imgDir 		= '%s/%s/' %(dataDir, dataSubType)
resultType  = args.result


num_id2prompt = {}
num_qu2prompt = {}
with open(f'number_prompt.txt') as number:
	for line in number:
		line = line.strip().split('\t')
		qid, quest, prompt = line[0], line[1].lower(), json.loads(line[2])
		# print(prompt)
		num_id2prompt[qid] = prompt
		num_qu2prompt[quest] = prompt


yn_id2prompt = {}
yn_qu2prompt = {}
with open(f'yesno_prompt.txt') as number:
	for line in number:
		line = line.strip().split('\t')
		qid, quest, prompt = line[0], line[1].lower(), json.loads(line[2])
		yn_id2prompt[qid] = prompt
		yn_qu2prompt[quest] = prompt


# Output file
resFile = f'vqa_dev_{args.prompt_type}_{args.portion}_m{args.mid}.json'

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)   
# class VQA:
# 	def __init__(self, annotation_file=None, question_file=None):
annIds = vqa.getQuesIds()
print('len of annIDs is', len(annIds))
anns = vqa.loadQA(annIds)
print('anns is', anns[:5])

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.clip, device, jit=False)

if args.few_shot:
	print(f"Loading model weights from {args.mount}/{args.checkpoint}{args.mid}.pt")
	if not args.finetuning:
		checkpoint = torch.load(f"{args.mount}/{args.checkpoint}{args.mid}.pt")
	else:
		checkpoint = torch.load(f"{args.mount}/DATA/MODELS/model_finetuning_epoch_{args.mid}.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.float()
	print("Model loaded")


results = []
yn_all = 0.0
yn_postive = 0.0
true_positive = 0.0
true_negative = 0.0
true_other = 0.0

num_all = 0.0
num_postive = 0.0
num_hund = 0.0
num_other = 0.0
pnoun = inflect.engine()

other_all = 0.0
other_positive = 0.0

vqa_prompt = {}
data_counter = 0
# with open(f'{args.mount}/DATA/{args.prompt_type}.json') as other:
# 	for line in other:
# 		data = json.loads(line.strip())
# 		vqa_prompt[data['qid']] = data
# 		data_counter += 1


with open(f'other_filtered_T5_large.json') as other:
	for line in other:
		data = json.loads(line.strip())
		vqa_prompt[data['qid']] = data
		data_counter += 1

with open(f'bad_cases_{args.prompt_type}_{args.portion}.txt','w+') as badcase, open(f'not_covered_{args.prompt_type}_{args.portion}.txt','w+') as notcovered:
	idx = len(anns)  // args.partition         #args.partition  64
	if args.portion == args.partition - 1:    
		anns = anns[idx * args.portion:]
	else:
		anns = anns[idx*args.portion: idx*(args.portion+1)]        #anns = vqa.loadQA(annIds)  len(3349)   args.portion 0    
	for ann in tqdm(anns):
		quesId = ann['question_id']
		imgId = ann['image_id']
		imgFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
		
		if os.path.isfile(imgDir + imgFilename):
			# Prepare image input
			image_input = preprocess(Image.open(imgDir + imgFilename)).unsqueeze(0).to(device)

			ground_answer = ann['multiple_choice_answer']

			# Prepare text inputs
			quesId = ann['question_id']
			quesVerify = str(quesId)
			question = vqa.qqa[quesId]['question'].lower().replace('?','').replace('.','')
			question_type = ann['question_type'].lower().strip(' ')+' '
			other_all += 1
			if quesVerify in vqa_prompt:
				prompt = vqa_prompt[quesVerify]['prompts'][0]
				classes = []
				for c in vqa_prompt[quesVerify]['labels']:
					if c not in ['[unk]','UNK','[UNK]']:
						classes.append(c)
				if args.oracle:
					if ground_answer not in classes:
						classes.append(ground_answer)
				text_inputs = torch.cat([clip.tokenize(f"{prompt.replace(args.mask_token, str(l))}") for l in classes]).to(device)
				# Calculate features
				with torch.no_grad():
					image_features = model.encode_image(image_input)
					text_features = model.encode_text(text_inputs)

				# Pick the top most similar labels for the image
				image_features /= image_features.norm(dim=-1, keepdim=True)
				text_features /= text_features.norm(dim=-1, keepdim=True)
				similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
				label_index = torch.argmax(similarity).cpu().numpy()
				results.append({"answer": classes[label_index], "question_id": quesId})

				if classes[label_index] == ann['answers'][0]['answer']:
					other_positive += 1
				else:
					print(f"{question}, {question_type}, {str(prompt)}, predict: {classes[label_index]}, ground truth: {ann['answers'][0]['answer']}, ID: {imgFilename}\n")
					badcase.write(f"{question}, {question_type}, {str(prompt)}, predict: {classes[label_index]}, ground truth: {ann['answers'][0]['answer']}, ID: {imgFilename}\n")
			else:
				results.append({"answer": "a", "question_id": quesId})
		else:
			raise FileNotFoundError

	else:
		results.append({"answer": "wrong answer type", "question_id": quesId})

	with open(resFile,'w+') as jsn:
		json.dump(results,jsn)
	print('result Json file has been written at %s.'%resFile)

	if args.yesno:
		print(f'YesNo: All: {yn_all}, Yes: {true_positive / yn_all * 100}, No: {true_negative / yn_all * 100}, Other: {true_other / yn_all * 100}, Acc:{yn_postive / yn_all * 100}')
	if args.number:
		print('Number: acc: %.2f, label covered: %.2f, other: %.2f' % (num_postive / num_all * 100, num_hund / num_all * 100, num_other / num_all * 100))
	if args.other:
		print('Other: acc: %.2f' % (num_postive / other_all * 100))
