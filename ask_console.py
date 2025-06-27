import psycopg2
import subprocess
from sentence_transformers import SentenceTransformer, util
import csv


def fetch_random_queries(connection, table_name, limit=5):
    """Retrieve random rows of (id, question, answer) from the database."""
    with connection.cursor() as cur:
        cur.execute(f"SELECT entry_id, question, answer FROM {table_name} ORDER BY random() LIMIT %s;", (limit,))
        return cur.fetchall()


def retrieve_top_contexts(connection, table_name, query_vec, top_k=3):
    """Fetch top_k context passages closest in embedding space to the query vector."""
    with connection.cursor() as cur:
        cur.execute(
            f"SELECT context FROM {table_name} ORDER BY embedding <=> %s::vector LIMIT {top_k};",
            (query_vec,)
        )
        return [row[0] for row in cur.fetchall()]


def generate_rag_response(prompt_text):
    """Run the local LLM in Docker to get a response for the given prompt."""
    try:
        proc = subprocess.Popen(
            ["docker", "exec", "-i", "rag_llm_service", "ollama", "run", "gemma:2b"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        out, err = proc.communicate(prompt_text)
        if proc.returncode != 0:
            return f"LLM Error: {err.strip()}"
        return out.strip()
    except Exception as err:
        return f"Docker Error: {err}"


def evaluate_models(db_config, table_name, output_path):
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)

    # Load embedding model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Gather random test samples
    samples = fetch_random_queries(conn, table_name)
    results = []

    for idx, (row_id, question, gold_answer) in enumerate(samples, 1):
        print(f"Processing ({idx}/{len(samples)}) - ID: {row_id}")

        # Embed the question
        q_vec = encoder.encode(question).tolist()

        # Retrieve contexts
        contexts = retrieve_top_contexts(conn, table_name, q_vec)
        combined_context = "\n\n".join(contexts)

        # Build enriched prompt
        prompt = (
            "You are an intelligent assistant.\n"
            "Here are some reference passages:\n\n"
            f"{combined_context}\n\n"
            f"Question: {question}\n"
            "Please answer succinctly and accurately."
        )

        # Get model response
        rag_reply = generate_rag_response(prompt)

        # Compute cosine similarity
        gold_emb = encoder.encode(gold_answer, convert_to_tensor=True)
        reply_emb = encoder.encode(rag_reply, convert_to_tensor=True)
        sim_score = util.cos_sim(gold_emb, reply_emb).item()

        # Record the outcome
        results.append({
            'id': row_id,
            'question': question,
            'expected': gold_answer,
            'generated': rag_reply,
            'similarity': round(sim_score, 4)
        })

    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation complete. Results saved to {output_path}")
    conn.close()


if __name__ == '__main__':
    database_params = {
        'host': 'localhost',
        'port': 5433,
        'dbname': 'rag_db',
        'user': 'rag_user',
        'password': 'rag_password'
    }
    evaluate_models(database_params, table_name='rag_entries', output_path='evaluation_results.csv')
