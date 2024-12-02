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
from factored_cognition_docstring import FactoredCognition
import yaml
from typing import Any

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

class Config:
    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation"""
        value = self.config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is None:
                return default
        return value

class APPSEvaluator:
    def __init__(self, config: Config):
        self.model = FactoredCognition(
            api_key=os.getenv("OPENAI_API_KEY"), 
            config=config
        )
        self.config = config
        self.semaphore = threading.Semaphore(config.get("execution", "num_concurrent", default=1))
        self.lock = threading.Lock()
        self.responses = {}
        self.save_dir = config.get("execution", "save_dir", default="./results")
        self.verbose = config.get("execution", "verbose", default=False)
        self.save_frequency = config.get("execution", "save_frequency", default=10)
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
                output = self.model.solve(prompt_text, index)
                output_str = json.dumps(output)
                # code_only = output.pythonCode
                if self.verbose:
                    print(f"API call successful for problem {index}")
            except Exception as e:
                raise e
            
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
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--smoketest',
        action='store_true',
        help='Only process the first problem for testing'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start processing from this problem index'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='Stop processing at this problem index (exclusive)'
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        help='List of specific problem indices to process'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args.config)
    
    problems = load_dataset("codeparrot/apps", "interview", split="test")
    
    if args.smoketest:
        problems = problems.select([0])
    elif args.indices:
        problems = problems.select(args.indices)
    else:
        end_index = args.end_index if args.end_index is not None else len(problems)
        problems = problems.select(range(args.start_index, end_index))
    
    evaluator = APPSEvaluator(config)
    
    try:
        with ThreadPoolExecutor(max_workers=config.get("execution", "num_concurrent", default=1)) as executor:
            futures = []
            for index, problem in enumerate(tqdm(problems)):
                # Keep the original index from the dataset
                original_index = args.indices[index] if args.indices else (args.start_index + index)
                future = executor.submit(evaluator.process_problem, original_index, problem)
                futures.append(future)
            
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        logging.info("Gracefully shutting down...")
        evaluator.save_results()
        raise
    
    evaluator.save_results()

if __name__ == "__main__":
    main()