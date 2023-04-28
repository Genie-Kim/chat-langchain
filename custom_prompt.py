from langchain.prompts.prompt import PromptTemplate

_template = """I want you to act like a Machine Learning engineer. Given the following conversation and a follow up question,
1. list up the terms that are logically related with question, and 2. rephrase the follow up question to be a standalone question.
Let's work this out in a step by step way to be sure we have the right answer.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""

prompt_template = """I want you to act like a Machine Learning paper reviewer. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say 3 keywords for Google search and don't try to make up an answer.
If you know the answer, organize just your answer using Markdown format with headings, subheadings, bullet points, and bold. Report the page number in the format as (Page #).

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
