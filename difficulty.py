#`This file used to generate uncertainty score for each question
# from utils import *
import time
import argparse
import numpy as np
import json
# from scipy.stats import entropy
# # from google.colab import files
# src = list(files.upload().values())[0]
# open('utils.py','wb').write(src)
# import utils

# import sys
# sys.path.insert(0,'/content/drive/MyDrive/active-prompt-main/paper/utils.py')
# # # sys.path.append('/content/drive/MyDrive/active-prompt-main/paper/utils.py')
from utils import *
import transformers
import os
import random
import pickle


def main():
    os.environ['CURL_CA_BUNDLE'] = ''
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    # print(f"API_KEY: {API_KEY}")
    set_random_seed(args.random_seed)
    
    # 첫 랜덤 1000개 뽑은 퀘스쳔 그대로 가져오기기
    with open('./uncertainty_result_gsm8k_k10.json', 'r') as f:
        json_data = json.load(f)
    
    idx_list = []
    for i in json_data:
        idx_list.append(i['dataset_idx'])
    dataloader = create_dataloader_our(args, idx_list)
    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    start =time.time()
    result, error_cnt = generate_difficulty_qes(args, dataloader)
    # end = time.time()
    # print('Total Execution Time: ', end - start, " seconds")
    # print(f'결과 : {result}')
    # print(f'에러 건 수 : {error_cnt}')
    # # output the results
    # path = f"./uncertainty_result_{args.dataset}_swiss_stage.pkl"
    # with open(path, 'wb') as f:
    #     pickle.dump(result, f)
    #     print('저장 잘됐다!')

def generate_difficulty_qes(args, dataloader):
    response_records = []
    
    for q in dataloader:
        if args.method == "zero_shot_cot":
            prompt = "Q: " + q['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]
        responses = llama3_request(model=args.model, input_prompt=prompt_list, temperature=args.temperature)
        pred_ans = answer_extraction(args, responses)
        score = 0
        response_record = {'question_idx':q['question_idx'], 'question': q['question'], 'responses':responses, 'difficulty_score':score}
        
        response_records.append(response_record)
        
    # 일반 토너먼트
    # final_questions, error_cnt = tournament(args, response_records)

    # 스위스 스테이지 토너먼트
    final_questions, error_cnt = swiss_stage_tournament(args, response_records)

    # # 난이도 점수 물어보기
    # final_questions, error_cnt = difficulty_answer(args, response_records)

    final_questions_idx_dict = []
    for i in final_questions:
        final_questions_idx_list.append(i['question_idx'])

    return final_questions_idx_list, error_cnt

import random

def difficulty_answer(args, questions):
    pass
    # our_prompt = "Please evaluate the difficulty of the following question.\n\
    #     The difficulty should be rated on a scale from 0 to 100, \
    #         where 0 represents the easiest question and 100 represents the most difficult question.\n\
    #             Your response must be strictly in the format: 'The difficulty of this question is {score}.\n\n"

    # for i in range(0, len(questions)):
    #     input_prompt = our_prompt + 
    

def swiss_stage_tournament(args, questions, rounds=10):
    our_prompt = "You will assess the difficulty of a question.\n\
        Specifically, given two pairs of a question, rationales, and an answer, you will determine which question is more difficult for you.\n\
            Respond with the question_idx of the more difficult question.\n\
            Your response must be strictly in the format: 'The more difficult idx of question is {question_idx}.\n\n"
    error_cnt = 0
    
    # Initialize scores for each question
    question_scores = {q['question_idx']: 0 for q in questions}
    
    for round_num in range(1, rounds + 1):
        print(f"Round {round_num}: {len(questions)} questions participating")
        
        # Sort questions by current score
        questions.sort(key=lambda q: question_scores[q['question_idx']], reverse=True)
        
        next_round = []
        
        for i in range(0, len(questions) - 1, 2):
            if question_scores[questions[i]['question_idx']] < -290 :
                # print(f'{round_num}라운드 {i}번째 팀부터는 안할거임')
                break

            input_prompt = our_prompt + "question_idx: " + str(questions[i]['question_idx']) + '\n' + \
                'Q: ' + questions[i]['question'] + '\n' + 'A: ' + questions[i]['responses'] + \
                    "\n\nquestion_idx: " + str(questions[i+1]['question_idx']) + '\n' + \
                        'Q: ' + questions[i+1]['question'] + '\n' + 'A: ' + questions[i+1]['responses']

            winner = llama3_request(model=args.model, input_prompt=input_prompt, temperature=args.temperature)
            winner_idx = answer_extraction(args, winner)
            print(f'이긴 인덱스 : {winner_idx}')

            if winner_idx != "":
                winner_li = [d for d in questions if d.get('question_idx') == int(winner_idx)]
                if len(winner_li) != 0:
                    if int(winner_idx) == questions[i]['question_idx']:
                        question_scores[questions[i]['question_idx']] += 1
                        question_scores[questions[i+1]['question_idx']] -= 100
                    else:
                        question_scores[questions[i]['question_idx']] -= 100
                        question_scores[questions[i+1]['question_idx']] += 1
                else:
                    error_cnt += 1
                    question_scores[questions[i]['question_idx']] += 1
                    question_scores[questions[i+1]['question_idx']] -= 100
            else:
                error_cnt += 1
                question_scores[questions[i]['question_idx']] += 1
                question_scores[questions[i+1]['question_idx']] -= 100

        if len(questions) % 2 == 1:
            question_scores[questions[-1]['question_idx']] -= 300
        
    # Sort by scores and return top 8 hardest questions
    hardest_questions = sorted(questions, key=lambda q: question_scores[q['question_idx']], reverse=True)[:8]
    return hardest_questions, error_cnt


def tournament(args, questions):
    round_num = 1
    our_prompt = "You will assess the difficulty of a question.\n\
        Specifically, given two pairs of a question, rationales, and an answer, you will determine which question is more difficult for you.\n\
            Respond with the question_idx of the more difficult question.\n\
            Your response must be strictly in the format: 'The more difficult idx of question is {question_idx}.'\n\n"
    error_cnt = 0
    while len(questions) > 8:
        print(f"Round {round_num}: {len(questions)} questions remaining")
        next_round = []
        random.shuffle(questions)
        
        winner_li = []
        for i in range(0, len(questions) - 1, 2):
            input_prompt = our_prompt + "question_idx: " + str(questions[i]['question_idx']) + '\n' + questions[i]['responses'] + "\n\nquestion_idx: " + str(questions[i+1]['question_idx']) + '\n' + questions[i+1]['responses']

            winner = llama3_request(model=args.model, input_prompt=input_prompt, temperature=args.temperature)
            winner = answer_extraction(args, winner)
            print(f'이긴 인덱스 : {winner}')
            if winner != "":
                winner = [d for d in questions if d.get('question_idx') == int(winner)]
                if len(winner) != 0:
                    next_round.append(winner[0])
                else:
                    error_cnt += 1
                    next_round.append(questions[i])
            else :
                error_cnt += 1
                next_round.append(questions[i])
        
        # If odd number of questions, last one automatically advances
        if len(questions) % 2 == 1:
            next_round.append(questions[-1])
        
        questions = next_round
        round_num += 1
    return questions, error_cnt

def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "last_letters", "strategyqa", "asdiv", "singleeq", "addsub", "multiarith, time_zone"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./basic_cot_prompts/math_word_problems", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--num_trails", type=int, default=10, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--sort_by", type=str, default='disagreement', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--concat_length", type=int, default=2, help='Used for task last_letters, indicates length of last letter concat'
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/train.jsonl" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "addsub":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/train.json" # train data path
        args.direct_answer_trigger = "\nThe answer is"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/train_rand_split.jsonl" # train data path
        args.direct_answer_trigger = "\nSo the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/train.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_train2.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'time_zone':
        args.dataset_path = "./dataset/timezone_convert/timezone_convertion_train.json"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


if __name__ == "__main__":
    main()