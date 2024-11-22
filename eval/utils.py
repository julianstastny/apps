from typing import Any, List, Dict
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from pydantic import BaseModel, create_model
from typing import Optional
from enum import Enum


class Client:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: str = None, verbose: bool = False, **kwargs):
        if not provider and not model:
            raise ValueError("Either provider or model must be specified")
        
        self.api_key = api_key
        self.verbose = verbose
        
        # Infer provider from model if not explicitly provided
        if model and not provider:
            if model.startswith(("gpt-", "ft:gpt-")):
                provider = "openai"
            elif model.startswith("claude-"):
                provider = "anthropic"
            else:
                raise ValueError(f"Cannot infer provider from model: {model}")
        
        self.provider = provider
        
        # Initialize client based on provider
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key, **kwargs)
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key, **kwargs)
        elif provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def __call__(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        response_format: Optional[BaseModel] = None,
        return_full_response: bool = False, 
        **kwargs
    ) -> Any:
        try:
            if self.provider == "anthropic":
                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    raise NotImplementedError("Structured output for Anthropic requires custom implementation")
                    
                if messages[0]['role'] == "system":
                    messages = messages[1:]
                    system_message = messages[0]['content']
                full_response = self.client.messages.create(model=model, system=system_message, messages=messages, **kwargs)
                if return_full_response:
                    return full_response
                else:
                    return full_response.content[0].text

            elif self.provider in ["openai", "openrouter"]:
                if self.provider == "openrouter" and response_format is not None:
                    raise ValueError("OpenRouter does not support response_format parameter")
                
                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    if self.provider == "openrouter":
                        raise ValueError("OpenRouter does not support response_format parameter")
                    full_response = self.client.beta.chat.completions.parse(
                        model=model, 
                        messages=messages, 
                        response_format=response_format,
                        **kwargs
                    )
                    message = full_response.choices[0].message
                    if message.refusal:
                        if self.verbose:
                            print(f"Model refused to respond: {message.refusal}")
                        return message.refusal
                    else:
                        return message.parsed
                else:
                    full_response = self.client.chat.completions.create(
                        model=model, 
                        messages=messages, 
                        **kwargs
                    )
                    if return_full_response:
                        return full_response
                    else:
                        return full_response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"API call failed: {str(e)}. Retrying...")
            raise



class Source(BaseModel):
    name: str
    url: str


class CodingSolution(BaseModel):
    reasoning: str
    pythonCode: str

class NewsArticle(BaseModel):
    title: str
    description: str
    content: str
    url: str
    image: str
    publishedAt: str
    source: Source

# class NewsArticleDiscriminator(BaseModel):
#     reasoningSteps: List[str]
#     articleAIsReal: bool
#     articleBIsReal: bool

# class InteractionAnalysis(BaseModel):
#     sanityChecks: str
#     generatorStrategy: str
#     discriminatorStrategy: str
#     additionalObservations: str
#     mainTakeaway: int
class DebateRound(BaseModel):
    Judge: str
    DebaterA: str
    DebaterB: str

# class DebateRoundDebaterBStarts(BaseModel):
#     Judge: str
#     DebaterB: str
#     DebaterA: str

class OpeningStatement(BaseModel):
    DebaterA: str
    DebaterB: str

# class ClosingStatement(BaseModel):
#     DebaterA: str
#     DebaterB: str

# class JudgeEvaluation(BaseModel):
#     finalDeliberation: str
#     AIsRight: bool
#     # BIsRight: bool

class Debate(BaseModel):
    openingStatement: OpeningStatement
    debateRounds: List[DebateRound]
    # closingStatement: ClosingStatement
    judgeFinalDeliberation: str
    AOverB: bool

class NewsDataLoader:
    def __init__(self, json_path: str):
        """Initialize the dataloader with path to JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data: List[Dict] = json.load(f)
        
    def __len__(self) -> int:
        """Return number of articles."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        """Return formatted article at given index."""
        article = self.data[idx]
        
        # Combine title, description and content with source
        formatted_text = (
            f"{article['title']}\n\n"
            f"{article['description']}\n\n" 
            f"{article['content']}\n\n"
            f"Source: {article['source']['name']}"
        )
        
        return formatted_text
    
    def get_all_articles(self) -> List[str]:
        """Return all articles in formatted form."""
        return [self[i] for i in range(len(self))]

# Example usage:
# loader = NewsDataLoader("gnews_top_headlines_trump_winner.json")
# first_article = loader[0]  # Get first article
# all_articles = loader.get_all_articles()  # Get all articles

def create_pydantic_models(model_configs: Dict[str, Dict]) -> Dict[str, BaseModel]:
    """Create Pydantic models dynamically from config definitions."""
    models = {}
    
    def get_field_type(type_str: str):
        """Convert type string to actual type, handling model references and Lists."""
        if type_str.startswith("List["):
            # Extract the inner type from List[Type]
            inner_type = type_str[5:-1]  # Remove 'List[' and ']'
            return List[get_field_type(inner_type)]
        if type_str in models:
            return models[type_str]
        return eval(type_str)
    
    # First create all models (needed for nested references)
    for model_name, fields in model_configs.items():
        models[model_name] = create_model(model_name)
    
    # Then fill in the fields
    for model_name, fields in model_configs.items():
        field_annotations = {
            field_name: (get_field_type(field_type), ...) 
            for field_name, field_type in fields.items()
        }
        models[model_name] = create_model(model_name, **field_annotations)
    
    return models

