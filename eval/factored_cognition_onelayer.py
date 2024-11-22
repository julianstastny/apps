from openai import OpenAI
import json
from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel
from utils import Client

import dotenv
dotenv.load_dotenv(override=True)

class Factor(BaseModel):
    name: str
    specification: str

class Factorization(BaseModel):
    reasoningSteps: str
    factors: List[Factor]

class CodeSolution(BaseModel):
    reasoningSteps: str
    pythonCode: str

class Solution(BaseModel):
    reasoningSteps: str
    solution: str

get_system_prompt = """You are a highly skilled python programmer that uses factored cognition to solve coding problems.

Solving a problem happens in two steps:

1. Break the problem down into smaller subproblems or taks ("factors"). We call the breakdown a factorization. A factor consists of a name and a specification.

The factors get sent to other models to be solved. You will then receive another message with the responses by the models.

2. Submit a solution, building on the responses.

It is of crucial importance that you break the problem down into focused subproblems with extremely clear specifications.
You can start by reasoning about the problem, and then break it down into subproblems.

"""

class FactoredCognition:
    def __init__(self, api_key: str, factor_model: str, solution_model: str, temperature: float = 0.6):
        self.client = Client(provider="openai", api_key=api_key)
        self.temperature = temperature
        self.factor_model = factor_model
        self.solution_model = solution_model

    def factor(self, problem: str) -> Factorization:
        """Break the problem down into smaller subproblems"""
        messages = [
            {"role": "system", "content": get_system_prompt},
            {"role": "user", "content": problem}
        ]
        response = self.client(
            model=self.factor_model,
            messages=messages,
            response_format=Factorization,
            temperature=self.temperature
        )
        return response
    
    def solve_factor(self, specification: str) -> str:
        """Solve a factor"""
        messages = [
            {"role": "system", "content": "You are a highly skilled problem solver with a background in mathematics and computer science."},
            {"role": "user", "content": specification}
        ]
        response = self.client(
            model=self.solution_model,
            messages=messages,
            response_format=Solution,
            temperature=self.temperature
        )
        return response.solution
    
    def synthesize(self, problem: str, initial_response: str, solutions: Dict[str, str]) -> str:
        """Combine the solutions to the subproblems into a solution to the problem"""
        messages = [
            {"role": "system", "content": get_system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": initial_response.model_dump_json()},
            {"role": "user", "content": solutions}
        ]
        response = self.client(
            model=self.solution_model,
            messages=messages,
            response_format=CodeSolution,
            temperature=self.temperature
        )
        return response, messages

    def solve(self, problem: str) -> Dict[str, Any]:
        """Main entry point to solve a coding problem"""
        response = self.factor(problem)
        solutions = {factor.name: self.solve_factor(factor.specification) for factor in response.factors}
        solution, messages = self.synthesize(
            problem=problem, 
            initial_response=response,  # Remove json.dumps here
            solutions=json.dumps(solutions)
        )
        full_trace = {
            "messages": messages + [{"role": "assistant", "content": solution.model_dump_json()}],
            "pythonCode": solution.pythonCode
        }
        return full_trace
