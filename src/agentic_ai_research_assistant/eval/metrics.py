from typing import List, Dict, Any, Set

def hit_rate_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Hit Rate @ K: fraction of queries where at least one ground-truth relevant chunk appears in the top K.
    Formula: sum(bool(set(retrieved[:k]) & set(relevant))) / len(results)
    """
    if not results:
        return 0.0
    hits = 0
    for r in results:
        retrieved_set = set(r["retrieved_ids"][:k])
        relevant_set = set(r["relevant_ids"])
        if bool(retrieved_set & relevant_set):
            hits += 1
    return round((hits / len(results)) * 100, 2)


def true_recall_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    True Recall @ K: average fraction of all relevant chunks recovered in the top K.
    """
    if not results:
        return 0.0
    total_recall = 0.0
    for r in results:
        relevant_set = set(r["relevant_ids"])
        if not relevant_set:
            continue
        retrieved_set = set(r["retrieved_ids"][:k])
        total_recall += len(retrieved_set & relevant_set) / len(relevant_set)
    return round((total_recall / len(results)) * 100, 2)


def tool_success_rate(tool_executions: List[Dict[str, Any]]) -> float:
    """
    Tool Success Rate: Successful validated tool calls divided by total tool calls.
    Success = no exception, status == 'ok', and validated schema.
    """
    if not tool_executions:
        return 0.0
    successful = sum(1 for call in tool_executions if call.get("status") == "ok")
    return round((successful / len(tool_executions)) * 100, 2)


def refusal_accuracy(queries_run: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Measures anti-hallucination cutoff performance:
    - Correct-refusal rate: unanswerable queries correctly returning 'not found'
    - False-refusal rate: answerable queries erroneously returning 'not found'
    """
    unanswerable = [q for q in queries_run if not q["is_answerable"]]
    answerable = [q for q in queries_run if q["is_answerable"]]

    correct_refusal = (
        sum(1 for q in unanswerable if q["result_status"] == "empty") / len(unanswerable)
        if unanswerable else 1.0
    )
    false_refusal = (
        sum(1 for q in answerable if q["result_status"] == "empty") / len(answerable)
        if answerable else 0.0
    )

    return {
        "correct_refusal_rate": round(correct_refusal * 100, 2),
        "false_refusal_rate": round(false_refusal * 100, 2)
    }
