"""
Run solutions from one problem.
"""
import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import time
import testing_util as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict

# At the very top of the file, right after imports
multiprocessing.set_start_method('spawn', force=True)

EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10

def get_report_path(save_dir: str, problem_index: int) -> str:
    """
    Constructs the path to the markdown report file.
    """
    reports_dir = os.path.join(save_dir, "reports")
    return os.path.join(reports_dir, f"tree_{problem_index}.md")

def print_aggregate_stats(results: Dict):
    """
    Print aggregate statistics to console only.
    """
    all_results = []
    for index in results:
        problem_results = np.asarray(results[index])
        if len(problem_results.shape) > 1:
            problem_results = problem_results[0]
        all_results.extend(problem_results)
    
    all_results = np.array(all_results)
    total_problems = len(results)
    
    # Aggregate statistics
    passed_problems = sum(np.all(np.asarray(results[idx]) > 0) for idx in results)
    test_case_avg = np.mean([np.mean(np.asarray(results[idx]) > 0) for idx in results])
    strict_accuracy = passed_problems / total_problems
    
    print(f"\nAggregate Statistics:")
    print(f"Test Case Average (average accuracy over problems) = {test_case_avg:.2f}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {strict_accuracy:.2f}")

def write_problem_results(results: Dict, args: argparse.Namespace):
    """
    Write individual problem results to markdown files.
    """
    for index in results:
        problem_results = np.asarray(results[index])
        if len(problem_results.shape) > 1:
            problem_results = problem_results[0]
        
        total = len(problem_results)
        compile_errors = np.sum(problem_results == -2)
        runtime_errors = np.sum(problem_results == -1)
        failed_tests = np.sum(problem_results == 0)
        passed_tests = np.sum(problem_results > 0)
        
        results_summary = [
            "Test case breakdown:",
            f"- ✅ Passed: {passed_tests}/{total} ({passed_tests/total:.1%})",
            f"- ❌ Failed: {failed_tests}/{total} ({failed_tests/total:.1%})",
            f"- ❌ (RE) Runtime Errors: {runtime_errors}/{total} ({runtime_errors/total:.1%})",
            f"- ❌ (CE) Compile Errors: {compile_errors}/{total} ({compile_errors/total:.1%})"
        ]

        report_path = get_report_path(args.save, index)
        if os.path.exists(report_path):
            with open(report_path, 'a') as f:
                f.write("\n\n## Test Results\n")
                f.write("\n".join(results_summary))

# Dummy `test_util.run_test` function for debugging multiprocessing.
def run_test(problem, test, debug):
    time.sleep(1)  # Simulate some work
    return [1]  # Dummy test result

def _temp_run(problem, generation, debug, result):
    try:
        if debug:
            print(f"Running test for problem: {problem}")
        result.append(test_util.run_test(problem=problem, test=generation, debug=debug))
        if debug:
            print(f"Test completed with result: {result}")
    except Exception as e:
        if debug:
            print(f"Error in _temp_run: {e}")

def check_correctness(problem, generation, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            print(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"Global timeout occurred, returning default result.")
    if debug:
        print(f"Final result: {result}")
    return result[0]


def eval_and_save_problems(args):
    problems = load_dataset("codeparrot/apps", split=f"{args.split}")

    codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    if args.start == 0 and args.end is None:
        results_loc = os.path.join(args.save, f"all_results.json")
    else:
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
    # print(codes_loc, results_loc)

    with open(codes_loc, "r") as f: 
        codes = json.load(f)

    # Only do the problems that are specified.
    if args.index:
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{args.index}]")
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{start}:{end}]")

    if args.stop_early:
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{start}:{args.stop_early}]")

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        try:
            if isinstance(codes, dict):
                code_entry = codes[str(index+args.start)]
                # Extract pythonCode if it's in JSON format
                if isinstance(code_entry, str):
                    try:
                        code_json = json.loads(code_entry)
                        # print(code_json)
                        output_strings = code_json.get("pythonCode", code_entry)
                        print(output_strings)
                    except json.JSONDecodeError:
                        output_strings = code_entry
                else:
                    output_strings = code_entry
            else:
                output_strings = codes[index+args.start]
        except:
            continue
        
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        sols = problem["solutions"]

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []
        if isinstance(output_strings, str):
            output_strings = [output_strings]
        for generation_idx, generation in enumerate(output_strings):
            if args.debug:
                print(f"\nTesting solution {generation_idx}, {generation=}")
            curr_res = [-2]
            try:
                curr_res = check_correctness(problem, generation=generation, timeout=TIMEOUT, debug=args.debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        if args.debug:
            print_aggregate_stats({index+args.start+args.index: res})

        results[index+args.start+args.index] = res
        
        # Always add test results to the markdown report
        write_problem_results({index+args.start+args.index: res}, args)
        
        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        results = {}
        if args.start == 0 and args.end is None:
            results_loc = os.path.join(args.save, f"all_results.json")
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
        
        if not os.path.exists(results_loc):
            print(f"No results file found at {results_loc}, exiting.")
            exit()

        with open(results_loc, "r") as f: 
            results = json.load(f)
        print_aggregate_stats(results)
        write_problem_results(results, args)
        exit()

    if not args.skip_evals:
        results = eval_and_save_problems(args)

    print_aggregate_stats(results)
    write_problem_results(results, args)


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("--skip_evals", action="store_true", help="If you want to skip the evals similar to print results.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--stop-early", default=None, type=int)
 
    args = parser.parse_args()

    # Set start method to 'spawn' for macOS compatibility
    # multiprocessing.set_start_method('spawn')

    main(args)