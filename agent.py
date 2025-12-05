# %% Minimal setup
# If needed (uncomment in a notebook):
#!pip install requests python-dotenv

import os, json, textwrap, re, time
import requests
import collections

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              


#------------------------------------------------------API CALL FUNCTION------------------------------------------------------#
def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}




#------------------------------------------------------------------------Inference-Time Techniques------------------------------------------------------------------------#
def technique_CoT(prompt: str) -> str:
    """
    Use Chain-of-Thought prompting to reason through math problems step-by-step.
    """

    print("Inside CoT technique")   #debug line

    #Cot implementation
    system_prompt = "You are a helpful math tutor. Think through the problem step-by-step but only provide the final answer, nothing else." #prompt encourages CoT reasoning

    call = call_model_chat_completions(prompt, system=system_prompt, temperature=0) #temperature 0 for deterministic output
    
    return (call["text"] or "").strip()

def technique_Self_Consistency(prompt: str, iterations: int = 5) -> str:
    """
    Use Self-Consistency by sampling multiple diverse reasoning paths and selecting the most consistent answer for common sense questions.
    """

    print("Inside self consistency technique")   #debug line

    #Self consistency implementation
    results = []
    
    for _ in range(iterations):
        call = call_model_chat_completions(prompt, temperature=0.7) #higher temperature for diversity
        result = (call["text"] or "").strip()
        results.append(result)
    
    count = collections.Counter(results)    #count occurrences of each answer
    top_result, _ = count.most_common(1)[0] #chooses top result based on most common answer

    return top_result

def technique_Verification(prompt):
    """
    Use Verification by generating a candidate answer and then verifying its correctness against given constraints for logic questions (i.e. multiple choice).
    """

    print("Inside verification technique")   #debug line

    #Verification implementation
    call = call_model_chat_completions(prompt)
    result = (call["text"] or "").strip()

    #check if multiple choice prompt
    if ("a." in prompt.lower() and "b." in prompt.lower() and "c." in prompt.lower()):
        for choice in ["a", "b", "c", "d"]:    #if the model's answer matches one of the choices, return that choice
            if result.strip().lower().startswith(choice) or choice in result.strip().lower():
                return choice

        call2 = call_model_chat_completions(prompt + "\n Answer with only one letter: a, b, c, d. Do not explain or elaborate.")  #else, ask model to pick a choice directly
        return (call2["text"] or "").strip()    #return the new answer
    
    #perhaps I will add new verification methods later

    return result

#--------------------------------------------------------Agent Loop--------------------------------------------------------#
def agent_loop(question_input: str) -> str:
    """
    Choose the Inference-Time Technique to apply based on the question_input.
    """
    # i can think of a lot of keywords for math problems, so I made an array to hold them all!
    math_keywords = [
        "solve", "calculate", " compute ", "equation",
        "how many", "total", "sum", "average", "percent",
        "ratio", "fraction", "integer", "distance", "ounces",
        "meters", "greater than", "less than", "minimum", "maximum", "shared", "divided", "multiplied", "added", "subtracted"
    ]

    # Simple heuristic to choose technique based on keywords in the question
    # Common sense questions
    if any(word in question_input.lower() for word in ["facts", "context", "plausible", "likely", "best describes"]):
        return technique_Self_Consistency(question_input)
    # math questions 
    elif any(word in question_input.lower() for word in math_keywords):
        return technique_CoT(question_input)
    # Default to logic questions
    else:
        return technique_Verification(question_input)