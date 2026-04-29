import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-7b-instruct")

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_BASE,
    temperature=0.0
)

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert linguist."),
    ("human", """
        Sentence: {text}
        Aspect: {aspect}

        Syntactic Dependency Sequence:
        {dependency_seq}

        Explanation (E): Each line in the sequence represents a word in the sentence, and each element within a line represents the ID, the word itself, its syntactic head, and its dependency relation.
        Based on the syntactic dependency of the sentence, analyze information related to the "{aspect}" in the sentence. Be concise and focus only on grammatical relations.
        """)
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting opinions."),
    ("human", """
        Sentence: {text}
        Aspect: {aspect}

        Syntactic Analysis: {step1_output}

        Considering the context and the syntactic information related to "{aspect}", what is the user's opinion towards "{aspect}"? 
        Identify the exact descriptive words if possible.
        """)
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analysis expert. Output ONLY ONE word: Positive, Negative, or Neutral."),
    ("human", """
        Opinion: {step2_output}

        Based on common sense and the speaker's opinion described above, what is the sentiment polarity towards "{aspect}"?
        Answer ONLY with exactly one of: Positive, Negative, or Neutral.
        """)
])

parser = StrOutputParser()

chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser
chain3 = prompt3 | llm | parser

from langsmith import traceable

@traceable(run_type="chain", name="Syn-Chain Pipeline")
def run_syn_chain(text: str, aspect: str, dependency_seq: str) -> dict:
    """
    Chạy 3 bước Syn-Chain để dự đoán cảm xúc của các khía cạnh.
    """
    step1_out = chain1.invoke({"text": text, "aspect": aspect, "dependency_seq": dependency_seq})

    step2_out = chain2.invoke({"text": text, "aspect": aspect, "step1_output": step1_out})

    step3_out = chain3.invoke({"aspect": aspect, "step2_output": step2_out})
    
    return {
        "step1_syntax": step1_out,
        "step2_opinion": step2_out,
        "step3_sentiment": step3_out.strip()
    }
