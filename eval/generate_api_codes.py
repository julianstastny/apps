import json
import os
import threading
from datasets import load_dataset
from tqdm import tqdm
import dotenv
from utils import Client, CodingSolution
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import datetime
import glob
from pathlib import Path

dotenv.load_dotenv(override=True)

def generate_prompt(test_case, prompt, starter_code=None):
    # _input = "\nQUESTION:\n"
    _input = prompt
    
    if starter_code:
        _input += "\n" + starter_code
        
    if not test_case.get("fn_name"):
        _input += "\n\nIMPORTANT: Use Standard Input format - your code should:\n"
        _input += "- Read input using input() or sys.stdin\n"
        _input += "- Print output to stdout using print()\n"
    else:
        _input += "\n\nIMPORTANT: Use Call-Based format - your code should:\n"
        _input += f"- Implement the function '{test_case['fn_name']}'\n"
        _input += "- Return values rather than printing them\n"
        _input += "- Not handle any input/output directly\n"
    return _input

class APPSEvaluator:
    def __init__(self, args):
        self.client = Client(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.semaphore = threading.Semaphore(args.num_concurrent)
        self.lock = threading.Lock()
        self.responses = {}
        self.save_dir = args.save_dir
        self.verbose = args.verbose
        self.model = args.model
        self.temperature = args.temperature
        self.system_prompt = args.system_prompt
        self.save_frequency = args.save_frequency
        os.makedirs(self.save_dir, exist_ok=True)

    def process_problem(self, index, problem):
        with self.semaphore:
            if self.verbose:
                print(f"\nProcessing problem {index}")
            
            try:
                problem["solutions"] = json.loads(problem["solutions"])
                problem["input_output"] = json.loads(problem["input_output"])
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for problem {index}: {e}")
                return
            
            if self.verbose:
                print(f"Generated prompt for problem {index}")
            
            prompt_text = generate_prompt(
                problem["input_output"],
                problem["question"],
                problem["starter_code"]
            )
            
            try:
                if self.verbose:
                    print(f"Calling API for problem {index}")
                output = self.client(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=self.temperature,
                    response_format=CodingSolution
                )
                output_str = output.model_dump_json()
                # code_only = output.pythonCode
                if self.verbose:
                    print(f"API call successful for problem {index}")
            except Exception as e:
                print(f"API error for index {index}: {e}")
                output_str = ""
            
            with self.lock:
                self.responses[index] = output_str
                if index % self.save_frequency == 0:
                    if self.verbose:
                        print(f"Saving results at checkpoint {index}")
                    self.save_results()

    def save_results(self):
        with open(os.path.join(self.save_dir, "all_codes.json"), "w") as f:
            json.dump(self.responses, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate code solutions using GPT API')
    parser.add_argument(
        '--num-concurrent',
        type=int,
        default=1,
        help='Number of concurrent API calls (default: 1)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    # New arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Base directory to save results (default: ./results)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='Model to use for generation (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Temperature for generation (default: 0.6)'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default='You are an expert python programmer. You are given coding problems. You first reason carefully about the problem, then you write the code to solve the problem.',
        help='System prompt for the model'
    )
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=10,
        help='Save results every N problems (default: 10)'
    )
    parser.add_argument(
        '--keep-runs',
        type=int,
        default=0,
        help='Number of previous runs to keep (default: 0, 0 for keeping all)'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load APPS dataset
    problems = load_dataset("codeparrot/apps", "interview", split="test")
    
    # Initialize evaluator with CLI-specified concurrency
    evaluator = APPSEvaluator(args)
    
    try:
        # Use ThreadPoolExecutor instead of raw threads
        with ThreadPoolExecutor(max_workers=args.num_concurrent) as executor:
            futures = []
            for index, problem in enumerate(tqdm(problems)):
                future = executor.submit(evaluator.process_problem, index, problem)
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        logging.info("Gracefully shutting down...")
        evaluator.save_results()
        raise
    
    # Final save
    evaluator.save_results()

if __name__ == "__main__":
    main()