from langchain.prompts.prompt import PromptTemplate

_template = """I want you to act like a machine learning engineer. Given the following conversation and a follow up question,
1. list up the relevant and precise terms, and 2. rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""

prompt_template = """I want you to act like a machine learning paper reviewer. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Format just your answer using Markdown. Use headings, subheadings, bullet points, and bold to well-organize the information.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)