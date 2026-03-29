import numpy as np
import re

import sys
sys.path.insert(0, "..")
from AI_agent.memory import get_high_quality_logs, count_logs
from config.config import (
    SIMILARITY_THRESHOLD,
    MAX_RESPONSE_LENGTH,
    MAX_DECOMPOSE_DEPTH,
    MAX_DB_RECORDS,
    MIN_SIMILARITY_DATA
)


def sentence_embedding(model, tokens):
    x = model.embedding.get_embeddings(tokens)
    x = model.block.forward(x)
    return np.mean(x, axis=0)


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)


def _get_logs_with_limit(min_rating=3):
    total = count_logs(min_rating=min_rating)
    if total > MAX_DB_RECORDS:
        limit = MAX_DB_RECORDS
    else:
        limit = total
    logs = get_high_quality_logs(min_rating=min_rating)
    return logs[:limit]


def predict_response_length(input_query, tokenizer, model, top_k=5):
    logs = _get_logs_with_limit(min_rating=3)
    if not logs:
        return None, []
    
    input_tokens = [tokenizer.encode(w) for w in input_query.split()]
    if not input_tokens:
        return None, []
    
    input_emb = sentence_embedding(model, input_tokens)
    
    similarities = []
    for query, response in logs:
        tokens = [tokenizer.encode(w) for w in query.split()]
        if not tokens:
            continue
        emb = sentence_embedding(model, tokens)
        sim = cosine_similarity(input_emb, emb)
        response_len = len(response.split())
        similarities.append((sim, response_len, query, response))
    
    if not similarities:
        return None, []
    
    max_sim = max(similarities, key=lambda x: x[0])
    
    if max_sim[0] >= SIMILARITY_THRESHOLD:
        return {
            "direct_response": max_sim[3],
            "max_similarity": max_sim[0],
            "source": "db_direct",
        }, []
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_similar = similarities[:top_k]
    
    total_weight = sum(s for s, _, _, _ in top_similar)
    if total_weight > 0:
        weighted_len = sum(s * l for s, l, _, _ in top_similar) / total_weight
    else:
        weighted_len = sum(l for _, l, _, _ in top_similar) / len(top_similar)
    
    word_count_estimate = int(weighted_len)
    
    return {
        "avg_similarity": sum(s for s, _, _, _ in top_similar) / len(top_similar),
        "estimated_word_count": word_count_estimate,
        "max_similarity": max(s for s, _, _, _ in top_similar),
        "source": "predicted",
    }, top_similar


def _parse_sub_questions(text):
    sub_questions = []
    lines = text.strip().split('\n')
    capture = False
    
    for line in lines:
        line = line.strip()
        if '##' in line.lower() and 'sub' in line.lower():
            capture = True
            continue
        if capture and line.startswith(('##', '#')):
            break
        if capture and line:
            match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if match:
                sub_questions.append(match.group(1).strip())
            elif not line.startswith(('-', '*', '•')):
                sub_questions.append(line)
    
    if not sub_questions:
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                sub_questions.append(cleaned)
    
    return sub_questions


def decompose_question(query, depth=0):
    if depth >= MAX_DECOMPOSE_DEPTH:
        return None
    
    prompt = f"""Decompose the following question into simple sub-questions. List each sub-question on a separate line.

Question: {query}

Requirements:
1. Break down into 3-8 simple sub-questions
2. Each sub-question should be self-contained and answerable in 1-2 sentences
3. Order them logically (prerequisites first)
4. Output ONLY the numbered list, nothing else

Example output:
1. What is X?
2. How does X work?
3. Why is X important?
"""
    
    response = cloud_agent(prompt)
    
    if not response or response.startswith("[Error"):
        return None
    
    sub_questions = _parse_sub_questions(response)
    
    if not sub_questions:
        return None
    
    return sub_questions


def _answer_sub_questions(sub_questions, tokenizer, model, depth=0):
    answers = []
    
    next_depth = depth + 1
    
    for sub_q in sub_questions:
        answer_text, source = generate_response(sub_q, tokenizer, model, depth=next_depth)
        answers.append({
            "question": sub_q,
            "answer": answer_text,
            "source": source
        })
    
    return answers


def _combine_answers(answers, original_query):
    if not answers:
        return ""
    
    prompt = f"""Combine the following Q&A pairs into a coherent answer for the original question.

Original Question: {original_query}

Q&A Pairs:
"""
    
    for i, item in enumerate(answers, 1):
        prompt += f"\nQ{i}: {item['question']}\nA{i}: {item['answer']}"
    
    prompt += """

Requirements:
1. Create a flowing, coherent response
2. Smoothly connect the answers
3. Maintain logical flow
4. Do not list Q&A separately - integrate into prose
"""
    
    combined = cloud_agent(prompt)
    return combined if combined else None


def _has_enough_data_for_similarity():
    return count_logs(min_rating=3) >= MIN_SIMILARITY_DATA


def generate_response(query, tokenizer, model, depth=0):
    if depth >= MAX_DECOMPOSE_DEPTH:
        return (
            "The question is too complex. Maximum recursion depth reached. "
            "Please try to simplify your question.",
            "max_depth_reached"
        )
    
    if not _has_enough_data_for_similarity():
        response = cloud_agent(query)
        return response, "cloud_direct"
    
    result, top_similar = predict_response_length(query, tokenizer, model)
    
    if result is None:
        response = cloud_agent(query)
        return response, "cloud_direct"
    
    if "direct_response" in result:
        return result["direct_response"], "db_direct"
    
    if result["estimated_word_count"] > MAX_RESPONSE_LENGTH:
        sub_questions = decompose_question(query, depth)
        
        if sub_questions:
            answers = _answer_sub_questions(sub_questions, tokenizer, model, depth)
            
            combined = _combine_answers(answers, query)
            
            if combined:
                return combined, "decomposed"
        
        response = cloud_agent(query)
        return response, "cloud_direct"
    
    response = cloud_agent(query)
    return response, "cloud_direct"


def cloud_agent(query):
    try:
        from config.config import OPENAI_API_KEY
        from openai import OpenAI
        if not OPENAI_API_KEY:
            return "[No API Key]"
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error: {str(e)}]"
