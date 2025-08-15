import streamlit as st
from langchain_core.prompts import PromptTemplate

# --- Evaluation Questions ---
EVALUATION_QUESTIONS = [
    {
        "question": "What is Logistic Regression?",
        "ground_truth": '''Logistic Regression is a supervised machine learning algorithm used for classification problems. 
                            Unlike linear regression which predicts continuous values it predicts the probability that an input belongs 
                            to a specific class. It is used for binary classification where the output can be one of two possible 
                            categories such as Yes/No, True/False or 0/1. It uses sigmoid function to convert inputs into a probability 
                            value between 0 and 1. In this article, we will see the basics of logistic regression and its core concepts.'''
    },
    {
        "question": "What is support vector machine",
        "ground_truth": '''
                            Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. 
                            It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when 
                            you want to do binary classification like spam vs. not spam or cat vs. dog.
                            The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better 
                            the model performs on new and unseen data.
                        '''
    },
    {
        "question": "What is decision tree?",
        "ground_truth": '''
                            A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. 
                            It works by splitting the data into subsets based on the value of input features. Each split is made 
                            to maximize the separation of classes or minimize the error in predictions.
                            The tree structure consists of nodes where each node represents a feature and branches represent the decision 
                            based on that feature. The leaves of the tree represent the final output or class label.
                        '''
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