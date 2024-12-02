from openai import OpenAI
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import os
from pydantic import BaseModel
from utils import Client
import networkx as nx
from datetime import datetime
from difflib import HtmlDiff, unified_diff
import yaml
from rich.console import Console
from rich.syntax import Syntax
import hashlib

import dotenv
dotenv.load_dotenv(override=True)

class Factor(BaseModel):
    name: str
    specification: str

class Factorization(BaseModel):
    reasoningSteps: str
    factors: List[Factor]

class CodeSolution(BaseModel):
    # reasoningSteps: str
    pythonCode: str

class Solution(BaseModel):
    # reasoningSteps: str
    solution: str

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

class CallTreeReporter:
    def __init__(self, output_dir: str = "reports"):
        self.problem = None
        self.final_solution = None
        self.chain_of_thought = None
        # List of (function_name, version, specification, solution) tuples
        self.factor_solutions = []
        self.output_dir = output_dir
        self.version_counter = {}  # Track versions per factor name
        os.makedirs(output_dir, exist_ok=True)
    
    def set_problem(self, problem: str):
        """Set the initial problem"""
        self.problem = problem
    
    def set_final_solution(self, solution: str):
        """Set the final solution"""
        self.final_solution = solution
    
    def set_chain_of_thought(self, cot: str):
        """Set the chain of thought reasoning"""
        self.chain_of_thought = cot
    
    def add_factor_solution(self, function_name: str, specification: str, solution: str):
        """Add a factor and its solution as a pair"""
        if function_name not in self.version_counter:
            self.version_counter[function_name] = 1
        else:
            self.version_counter[function_name] += 1
        
        self.factor_solutions.append((
            function_name,
            self.version_counter[function_name],
            specification,
            solution
        ))
    
    def save(self, problem_hash: str) -> str:
        """Generate and save markdown report"""
        filename = f"{self.output_dir}/tree_{problem_hash}.md"
        
        with open(filename, 'w') as f:
            # Write problem
            f.write(f"# Problem\n{self.problem}\n\n")
            
            # Write chain of thought if it exists
            if self.chain_of_thought:
                f.write("## Chain of Thought\n")
                f.write(f"{self.chain_of_thought}\n\n")
            
            # Write factors and their solutions
            f.write("## Sub-solutions\n\n")
            for function_name, version, spec, solution in self.factor_solutions:
                f.write(f"### {function_name} (Iteration {version})\n")
                f.write(f"#### Specification\n{spec}\n\n")
                f.write(f"#### Implementation\n```python\n{solution}\n```\n\n")
            
            # Write final solution
            f.write(f"\n## Final Solution\n```python\n{self.final_solution}\n```\n")
        
        return filename

class FactoredCognition:
    def __init__(self, api_key: str, config: Config):
        self.client = Client(provider="openai", api_key=api_key)
        self.temperature = config.get("models", "temperature", default=0.6)
        self.factor_model = config.get("models", "factor_model")
        self.solution_model = config.get("models", "solution_model")
        self.system_prompts = config.get("system_prompts")
        self.verbose = config.get("execution", "verbose", default=False)
        config_hash = hashlib.sha256(json.dumps(config.config, sort_keys=True).encode()).hexdigest()[:16]
        self.output_dir = config.get("execution", "save_dir", default=f"results_{config_hash}")
        self.console = Console()
        self.tools = config.get("tools")
        self.max_recursion_depth = config.get("models", "max_recursion_depth", default=3)
        os.makedirs(self.output_dir + "/reports", exist_ok=True)

    def _get_tools(self, first_iteration: bool = False, last_iteration: bool = False):

        cot_tool = {
            "type": "function",
            "function": {
                "name": "reason_about_problem",
                "description": "Call this to reason about the right abstraction of the problem before submitting specifications.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chain_of_thought": {
                            "type": "string",
                            "description": "A chain of thought reasoning about the right abstraction of the problem"
                        }
                    },
                    "required": ["chain_of_thought"]
                }
            },
            "strict": True
        }

        spec_tool = {
            "type": "function",
            "function": {
                "name": "submit_specification",
                "description": self.tools["specification"]["description"],
                "parameters": {
                    "type": "object",
                    "properties": self.tools["specification"]["properties"],
                    "required": self.tools["specification"]["required"]
                }
            },
            "strict": True
        }
        
        solution_tool = {
            "type": "function",
            "function": {
                "name": "submit_solution",
                "description": self.tools["solution"]["description"],
                "parameters": {
                    "type": "object",
                    "properties": self.tools["solution"]["properties"],
                    "required": self.tools["solution"]["required"]
                }
            },
            "strict": True
        }
        
        if first_iteration:
            return [cot_tool, spec_tool]
        elif last_iteration:
            return [solution_tool]
        else:
            return [spec_tool, solution_tool]
    
    def solve_factor(self, specification: str) -> str:
        """Solve a factor"""
        messages = [
            {"role": "system", "content": self.system_prompts["solution"]},
            {"role": "user", "content": specification}
        ]
        response = self.client(
            model=self.solution_model,
            messages=messages,
            response_format=CodeSolution,
            temperature=self.temperature,
        )
        return response.pythonCode
    

    def solve(self, problem: str, index: int = None) -> Dict[str, Any]:
        """Main entry point to solve a coding problem"""
        if index is None:
            problem_id = hashlib.sha256(problem.encode()).hexdigest()[:16]
        else:
            problem_id = index
        
        reporter = CallTreeReporter(output_dir=f"{self.output_dir}/reports")
        reporter.set_problem(problem)
        
        messages = [
            {"role": "system", "content": self.system_prompts["main"].format(max_recursion_depth=self.max_recursion_depth)},
            {"role": "user", "content": problem}
        ]

        all_messages, final_code, sub_solutions = self.iterate(
            messages, 
            sub_solutions={},
            reporter=reporter
        )
        
        reporter.set_final_solution(final_code)
        report_path = reporter.save(problem_id)
        
        combined_code = "\n\n".join(list(sub_solutions.values()) + [final_code])
        
        full_trace = {
            "messages": all_messages,
            "submittedSolution": final_code,
            "pythonCode": combined_code,
            "report_path": report_path
        }

        if self.verbose:
            syntax = Syntax(combined_code, "python", theme="monokai")
            self.console.print(syntax)
        
        return full_trace

    def iterate(self, messages: List[Dict[str, str]], sub_solutions: Dict[str, str], 
                recursion_depth: int = 0, reporter: Optional[CallTreeReporter] = None) -> Tuple[List[Dict[str, str]], str, Dict[str, str]]:
        """Iterate on the solution to the problem"""
        last_iteration = False
        if recursion_depth == 0:
            tools = self._get_tools(first_iteration=True)
            message_appendix = ""
        elif recursion_depth == self.max_recursion_depth:
            tools = self._get_tools(last_iteration=True)
            last_iteration = True
            message_appendix = "Maximum number of iterations reached. You are required to submit a solution now."
        else:
            tools = self._get_tools()
            message_appendix = f"You can iterate on the functions for up to {self.max_recursion_depth - recursion_depth} more times. But if you are happy with the implementations of all functions, you can submit a solution now. And in case you are not happy with some of them, remember that you only need to iterate on functions that you are not happy with: the others are saved and will be used when you submit the solution."

        if message_appendix:
            messages.append({"role": "user", "content": message_appendix})
        
        response = self.client(
            model=self.factor_model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            tool_choice="required",
            return_full_response=True,
            parallel_tool_calls=not last_iteration
        )

        print(response.choices[0].message.model_dump_json(indent=2))
        tool_calls = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message.model_dump())

        if len(tool_calls) == 1:
            if tool_calls[0].function.name == "submit_solution":
                return messages, json.loads(tool_calls[0].function.arguments)["pythonCode"], sub_solutions
            elif tool_calls[0].function.name == "reason_about_problem":
                args = json.loads(tool_calls[0].function.arguments)
                if reporter is not None:
                    reporter.set_chain_of_thought(args["chain_of_thought"])
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].id,
                    "content": "Thanks for submitting your chain of thought. You can now submit specifications."
                    }
                messages.append(tool_message)
                return self.iterate(messages, sub_solutions, recursion_depth, reporter)
            elif tool_calls[0].function.name == "submit_specification" and recursion_depth == 0:
                return messages, "# Only one specification submitted.", {}
        
        for tool_call in tool_calls:
            if tool_call.function.name == "submit_solution":
                raise ValueError("submit_solution called along with other tools")
        
        tool_messages = []

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            sub_result = self.solve_factor(json.dumps(args))
            
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": sub_result
            })
            
            sub_solutions[args["function_name"]] = sub_result
            
            if reporter is not None:
                reporter.add_factor_solution(args["function_name"], args["docstring"], sub_result)

        messages.extend(tool_messages)
        
        return self.iterate(messages, sub_solutions, recursion_depth + 1, reporter)
