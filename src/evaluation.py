import streamlit as st
from langchain_core.prompts import PromptTemplate

# --- Evaluation Questions ---
EVALUATION_QUESTIONS = [
    {
        "question": "What is the Name on the document?",
        "ground_truth": "Shashwat krishna Dwiwedi"
    },
    {
        "question": "What is the aadhar card number",
        "ground_truth": "301388714810"
    }
]

def get_evaluation_score(question, generated_answer, ground_truth, llm):
    """
    Uses an LLM to score the generated answer against the ground truth.
    """
    try:
        eval_prompt_template = """
        You are an impartial evaluator. Your task is to assess the quality of a generated answer based on a ground truth reference.
        Provide a score from 1 to 10, where 1 is poor and 10 is excellent, based on accuracy and relevance. Also, provide a brief justification for your score.

        Here is the data:
        [Question]: {question}
        [Ground Truth]: {ground_truth}
        [Generated Answer]: {generated_answer}

        Please provide your response in the following format, and nothing else:
        Score: [Your score from 1 to 10]
        Justification: [Your brief justification]
        """
        
        eval_prompt = PromptTemplate.from_template(eval_prompt_template)
        
        # This is a simple chain that takes the formatted prompt and sends it to the LLM
        eval_chain = eval_prompt | llm
        
        response_content = eval_chain.invoke({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer
        }).content

        # Parse the score and justification from the LLM's response
        score_line = [line for line in response_content.split('\n') if 'Score:' in line][0]
        justification_line = [line for line in response_content.split('\n') if 'Justification:' in line][0]
        
        score = int(score_line.split(':')[1].strip())
        justification = justification_line.split(':')[1].strip()
        
        return score, justification

    except Exception as e:
        st.error(f"Error during evaluation scoring: {e}")
        return 0, "Error in scoring"