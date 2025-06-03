import pandas as pd
import argparse
import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='combine')
    parser.add_argument('--model', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--temperature', type=int, default=0.75)
    parser.add_argument('--top_p', type=int, default=0.9)
    parser.add_argument('--max_tokens', type=int, default=1024)
    args = parser.parse_args()
    os.makedirs('in_context_eval', exist_ok=True)


    def convert_dict_list_to_string(dict_list):
        """
        Convert a list of dictionaries to a formatted string.
        """
        result = ""
        for i, item in enumerate(dict_list, 1):
            for instruction, response in item.items():
                result += f"{i}. Instruction: {instruction}\nResponse: {response}\n"
        return result.strip()


    def generate_predict_vanilla():
        with open("prompt_template/in_context_prompt_vanilla", 'r') as file:
            instruction = file.read()

        llm = LLM(model=args.model, gpu_memory_utilization=0.9, tensor_parallel_size=args.num_gpu)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        df = pd.read_json("../dataset/in_context_eval.json")
        close_ended_list = df["closed_ended_question"].tolist()
        domain_list = df["domain"].tolist()
        answer_list = df["answer"].tolist()

        prompts = []
        for i in range(len(df)):
            formatted_data = []
            formatted_data.append({"role": "user", "content": instruction.format(domain_list[i], close_ended_list[i])})
            prompts.append(tokenizer.apply_chat_template(formatted_data, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        f_outputs = [output.outputs[0].text for output in outputs]
        df["pred_answer"] = f_outputs
        df = df.drop('open_ended', axis=1)

        records = df.to_dict(orient='records')
        with open('in_context_eval/in_context_eval_vanilla_{}.json'.format(args.model), 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))

        return answer_list, f_outputs



    def generate_predict_out_topic():
        with open("prompt_template/in_context_prompt", 'r') as file:
            instruction = file.read()

        llm = LLM(model=args.model, gpu_memory_utilization=0.9, tensor_parallel_size=args.num_gpu)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        df = pd.read_json("../dataset/in_context_eval_outtopic.json")
        close_ended_list = df["closed_ended_question"].tolist()
        open_ended_list = df["instruction_pair"].tolist()
        domain_list = df["domain"].tolist()
        answer_list = df["answer"].tolist()

        prompts = []
        for i in range(len(df)):
            formatted_data = []
            formatted_data.append({"role": "user", "content": instruction.format(domain_list[i], domain_list[i], domain_list[i], convert_dict_list_to_string(open_ended_list[i]), close_ended_list[i])})
            prompts.append(tokenizer.apply_chat_template(formatted_data, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        f_outputs = [output.outputs[0].text for output in outputs]
        df["pred_answer"] = f_outputs
        df = df.drop('open_ended', axis=1)

        records = df.to_dict(orient='records')
        with open('in_context_eval/in_context_eval_outtopic_{}.json'.format(args.model), 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))

        return answer_list, f_outputs


    def generate_predict_subreddit():
        with open("prompt_template/in_context_prompt_subreddit", 'r') as file:
            instruction = file.read()

        llm = LLM(model=args.model, gpu_memory_utilization=0.9, tensor_parallel_size=args.num_gpu)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        df = pd.read_json("../dataset/in_context_eval.json")
        close_ended_list = df["closed_ended_question"].tolist()
        domain_list = df["domain"].tolist()
        subreddit_list = df["subreddit"].tolist()
        answer_list = df["answer"].tolist()

        prompts = []
        for i in range(len(df)):
            formatted_data = []
            formatted_data.append({"role": "user", "content": instruction.format(domain_list[i], subreddit_list[i], close_ended_list[i])})
            prompts.append(tokenizer.apply_chat_template(formatted_data, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        f_outputs = [output.outputs[0].text for output in outputs]
        df["pred_answer"] = f_outputs
        df = df.drop('open_ended', axis=1)

        records = df.to_dict(orient='records')
        with open('in_context_eval/in_context_eval_subreddit_{}.json'.format(args.model), 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))

        return answer_list, f_outputs


    def generate_predict_in_topic():
        with open("prompt_template/in_context_prompt", 'r') as file:
            instruction = file.read()

        llm = LLM(model=args.model, gpu_memory_utilization=0.9, tensor_parallel_size=args.num_gpu)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        df = pd.read_json("../dataset/in_context_eval.json")
        close_ended_list = df["closed_ended_question"].tolist()
        open_ended_list = df["open_ended"].tolist()
        domain_list = df["domain"].tolist()
        answer_list = df["answer"].tolist()

        prompts = []
        for i in range(len(df)):
            formatted_data = []
            formatted_data.append({"role": "user", "content": instruction.format(domain_list[i], domain_list[i], domain_list[i], convert_dict_list_to_string(open_ended_list[i]), close_ended_list[i])})
            prompts.append(tokenizer.apply_chat_template(formatted_data, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        f_outputs = [output.outputs[0].text for output in outputs]
        df["pred_answer"] = f_outputs
        df = df.drop('open_ended', axis=1)

        records = df.to_dict(orient='records')
        with open('in_context_eval/in_context_eval_intopic_{}.json'.format(args.model), 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))

        return answer_list, f_outputs

    def generate_predict_combine():
        with open("prompt_template/in_context_prompt", 'r') as file:
            instruction = file.read()

        llm = LLM(model=args.model, gpu_memory_utilization=0.9, tensor_parallel_size=args.num_gpu)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        df = pd.read_json("../dataset/in_context_eval.json")
        close_ended_list = df["closed_ended_question"].tolist()
        open_ended_list = df["open_ended"].tolist()
        domain_list = df["domain"].tolist()
        subreddit_list = df["subreddit"].tolist()
        answer_list = df["answer"].tolist()

        prompts = []
        for i in range(len(df)):
            formatted_data = []
            formatted_data.append({"role": "user", "content": instruction.format(domain_list[i], subreddit_list[i], domain_list[i], convert_dict_list_to_string(open_ended_list[i]), close_ended_list[i])})
            prompts.append(tokenizer.apply_chat_template(formatted_data, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)
        f_outputs = [output.outputs[0].text for output in outputs]
        df["pred_answer"] = f_outputs
        df = df.drop('open_ended', axis=1)

        records = df.to_dict(orient='records')
        with open('in_context_eval/in_context_eval_combine_{}.json'.format(args.model), 'w') as json_file:
            json_file.write(json.dumps(records, indent=4))

        return answer_list, f_outputs

    def extract_letters(input_list):
        """
        Extract letters before the period in each list item.

        Args:
            input_list (list): List of strings with format like "A. Some text"

        Returns:
            list: List of extracted letters
        """
        extracted_letters = []
        for item in input_list:
            # Split the string and take the first part (before the period)
            letter = item.split('.')[0].strip()
            extracted_letters.append(letter)
        return extracted_letters


    def calculate_accuracy(actual, predicted):
        """
        Calculate accuracy from two lists of values.
        Considers a prediction correct if:
        1. The prediction exactly matches the actual value, OR
        2. The predicted string is contained within the actual string

        Parameters:
        actual (list): List of actual/true values
        predicted (list): List of predicted values

        Returns:
        float: Accuracy as a percentage
        """
        if len(actual) != len(predicted):
            raise ValueError("Both lists must have the same length")

        correct = 0
        for a, p in zip(actual, predicted):
            # Convert values to strings to handle all data types
            a_str = str(a)
            p_str = str(p)

            # Count as correct if exact match or if predicted is contained in actual
            if a_str == p_str or p_str in a_str:
                correct += 1

        total = len(actual)
        accuracy = (correct / total)

        return accuracy

    if args.config == "vanilla":
        true_answers, raw_pred_answers = generate_predict_vanilla()
        predicted_answers = extract_letters(raw_pred_answers)
        accuracy = calculate_accuracy(true_answers, predicted_answers)
        print(f"Accuracy: {accuracy:.3f}")
    elif args.config == "out_topic":
        true_answers, raw_pred_answers = generate_predict_out_topic()
        predicted_answers = extract_letters(raw_pred_answers)
        accuracy = calculate_accuracy(true_answers, predicted_answers)
        print(f"Accuracy: {accuracy:.3f}")
    elif args.config == "subreddit":
        true_answers, raw_pred_answers = generate_predict_subreddit()
        predicted_answers = extract_letters(raw_pred_answers)
        accuracy = calculate_accuracy(true_answers, predicted_answers)
        print(f"Accuracy: {accuracy:.3f}")
    elif args.config == "in_topic":
        true_answers, raw_pred_answers = generate_predict_in_topic()
        predicted_answers = extract_letters(raw_pred_answers)
        accuracy = calculate_accuracy(true_answers, predicted_answers)
        print(f"Accuracy: {accuracy:.3f}")
    elif args.config == "combine":
        true_answers, raw_pred_answers = generate_predict_combine()
        predicted_answers = extract_letters(raw_pred_answers)
        accuracy = calculate_accuracy(true_answers, predicted_answers)
        print(f"Accuracy: {accuracy:.3f}")
    else:
        print("Please choose right config.")

