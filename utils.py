# This file contains necessary helper functions
# e.g. llama3_request, create_dataloader
import random
import sys
import numpy as np
import torch
import json
import re
from collections import Counter
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
from dotenv import load_dotenv
import os

# .env 파일 활성화
load_dotenv()
hf_KEY = os.getenv('HF_API_KEY')


#**랜덤 시드값 설정**
# set the random seed for reproducibility
def set_random_seed(seed):    #랜덤 숫자 만들 때 사용할 숫자 지정
    random.seed(seed)
    np.random.seed(seed)   #넘파이에서도 같은 숫자 사용
    torch.manual_seed(seed) #파이토치에서도 같은 숫자 사용
    if torch.cuda.is_available():   #GPU 사용 
        torch.cuda.manual_seed_all(seed)

#**모델 설정**
def llama3_request(model:str, input_prompt:list, temperature=0.7):
    messages = [{"role": "user", "content": input_prompt}]
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        token = hf_KEY
    )

    # print('추론 시작!')  
    outputs = pipeline(
        messages,
        max_new_tokens=256
    )
    outputs = outputs[0]["generated_text"][-1]['content']
    # print(f'생성 텍스트 : {outputs}')
    # print('추론 끝!')

    return outputs


#**데이터셋 불러오고 질문/정답 리스트로 저장 설정**
def load_data(args):
    questions = []   #질문 리스트 초기화, 각 데이터셋에서 추출한 질문 저장 
    answers = [] #답변 리스트 초기화, 각 데이터셋에서 추출한 답변 저장 
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":   #데이터셋 이름 
        with open(args.dataset_path) as f:   #데이터셋 파일 경로 열기 
            lines = f.readlines()             # 파일에서 모든 줄 읽어 lines라는 리스트에 저장 
            for line in lines:                #각 줄에 대해 처리 
                json_res = decoder.raw_decode(line)[0]      #각 줄을 json 형식으로 변환해 json_res 저장 
                questions.append(json_res["question"].strip())    #questions빈 리스트에 어펜드 (json_res의 question 부분 공백 제거 후)
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))   #json_res의 answer부분 읽고 #### 이후 부분 골라 가져오고 , 기호는 없애서 answers 리스트에 어펜드 
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    elif args.dataset == 'time_zone':
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line['question'].strip()
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


#**질문/정답 딕셔너리로 , 데이터 순서 섞기 설정**
# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset

def create_dataloader_our(args, idx_list)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []             # 빈 리스트를 만들어서 데이터를 저장할 준비
    for idx in range(len(questions)):
        if idx in idx_list:
            dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


#**LLM 입력 전 기존 cot 프롬프트 가져오기 설정**
# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:               #입력 프롬프트  #cot_flag: chain of thought ,True인 경우 모델이 사고과정을 같이 설명하며 답을 제공하는 프롬프트 생성 
    x, z, y = [], [], []                                         #x: 질문, z: rationale ,y: 정답 빈 리스트 지정 
    
    with open(args.prompt_path, encoding="utf-8") as f:           #prompt_path: 기존 CoT 프롬프트 경로 utf-8로 인코딩해서 열기 
        json_data = json.load(f)                                  ## JSON 형식의 데이터를 불러옴
        json_data = json_data["prompt"]                            # "prompt"라는 키의 값만 가져옴
        for line in json_data:
            x.append(line["question"])                            #한 줄씩 순차적으로 읽어 리스트에 어펜드 
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))                               #x(question) 길이에 맞춰 인덱스 리스트 생성 
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text                                                                    #완성 프롬프트 반환


#모델이 뱉을 때는 질문,응답 같이 뱉어서 후처리 필요 
##첫번째 LLM이 뱉은 응답 추출 
def answer_extraction(args, responses):          #response에 값을 넣으면 response
    pred_ans = ""                      # 정답을 저장할 변수 초기화
    temp = ""                           # 응답을 임시 저장할 변수 초기화
    if args.model == 'gpt-3.5-turbo':                                #해당 없음 ->llama_request return에서 지정해줌
        temp = responses['choices'][0]['message']['content']
    else:
        temp = responses                        #텍스트 응답 가져옴  ->llama_request return에서 지정해줌
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")                          #숫자만 추출하도록 처리
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    # elif args.dataset in ("aqua", "csqa"):
    #     temp = re.findall(r'A|B|C|D|E', temp)
    # elif args.dataset in ("strategyqa", "coin_flip"):
    #     temp = temp.lower()
    #     temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
    #     temp = temp.split(" ")
    #     temp = [i for i in temp if i in ("yes", "no")]
    # elif args.dataset in ("last_letters"):
    #     temp = re.sub("\"|\'|\n|\.|\s","", temp)
    #     temp = [temp]
    # elif args.dataset in ('time_zone'):
    #     temp = temp.split('The answer is ')[-1].replace('.', '')
    #     temp = [temp]

    if len(temp) != 0:      # 추출된 값이 비어 있지 않으면 정답을 추출 (temp:후처리된 응답)
        answer = temp[-1]        # 가장 마지막 값을 정답으로 설정         ?   (cot로 숫자들이 연속되서 마지막 값이 정답)
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":            # 정답 끝에 점(.)이 있으면 제거
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))            # 실수로 변환 후 반올림
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""                          # 오류가 나면 정답을 빈 문자열로 설정
        pred_ans = answer                # 정답을 pred_ans에 저장
    else:
        pred_ans = ""
    return pred_ans


def find_most_frequent(arr, n): # 주어진 배열의 처음 n개의 요소에서 가장 많이 나오는 항목을 찾기
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])               #배열에서 각 항목의 빈도를 세는 Counter 사용
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]     # 가장 자주 등장한 항목과 그 빈도수 찾기
    return frequency, most_frequent_item