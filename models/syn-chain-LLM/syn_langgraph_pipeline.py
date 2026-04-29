from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

QWEN_API_BASE = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-7b-instruct")

llm = ChatOpenAI(model=MODEL_NAME, api_key=QWEN_API_KEY, base_url=QWEN_API_BASE, temperature=0.0)

prompt1 = ChatPromptTemplate.from_messages([("system", "You are an expert linguist."), ("human", "Sentence: {text}\nAspect: {aspect}\n\nSyntactic Dependency Sequence:\n{dependency_seq}\n\nBased on the syntactic dependency of the sentence, analyze information related to the \"{aspect}\" in the sentence.")])
prompt2 = ChatPromptTemplate.from_messages([("system", "You are an expert at extracting opinions."), ("human", "Sentence: {text}\nAspect: {aspect}\n\nSyntactic Analysis: {step1_output}\n\nConsidering the context and the syntactic information related to \"{aspect}\", what is the user's opinion towards \"{aspect}\"?")])
prompt3 = ChatPromptTemplate.from_messages([("system", "You are a sentiment analysis expert. Output ONLY ONE word: Positive, Negative, or Neutral."), ("human", "Opinion: {step2_output}\n\nBased on common sense and the speaker's opinion described above, what is the sentiment polarity towards \"{aspect}\"?\nAnswer ONLY with exactly one of: Positive, Negative, or Neutral.")])

parser = StrOutputParser()
chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser
chain3 = prompt3 | llm | parser

class AgentState(TypedDict):
    text: str
    aspect: str
    dependency_seq: str
    step1_output: str
    step2_output: str
    step3_output: str

def syntax_analysis_node(state: AgentState):
    return {"step1_output": chain1.invoke({"text": state["text"], "aspect": state["aspect"], "dependency_seq": state["dependency_seq"]})}

def opinion_extraction_node(state: AgentState):
    return {"step2_output": chain2.invoke({"text": state["text"], "aspect": state["aspect"], "step1_output": state.get("step1_output", "")})}

def sentiment_prediction_node(state: AgentState):
    return {"step3_output": chain3.invoke({"aspect": state["aspect"], "step2_output": state.get("step2_output", "")})}

workflow = StateGraph(AgentState)

workflow.add_node("syntax_analysis", syntax_analysis_node)
workflow.add_node("opinion_extraction", opinion_extraction_node)
workflow.add_node("sentiment_prediction", sentiment_prediction_node)

workflow.set_entry_point("syntax_analysis")
workflow.add_edge("syntax_analysis", "opinion_extraction")
workflow.add_edge("opinion_extraction", "sentiment_prediction")
workflow.add_edge("sentiment_prediction", END)

graph = workflow.compile()
