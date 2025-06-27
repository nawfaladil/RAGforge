import psycopg2
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def initialize_database(connection):
    """Set up the vector extension and table schema."""
    with connection.cursor() as cur_pg:
        cur_pg.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur_pg.execute("DROP TABLE IF EXISTS rag_entries;")
        cur_pg.execute(
            """
            CREATE TABLE rag_entries (
                entry_id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                context TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding vector(384)
            );
            """
        )
    connection.commit()


def ingest_samples(connection, dataset, encoder):
    """Embed each context and insert into the database."""
    with connection.cursor() as cur_pg:
        for idx, record in enumerate(dataset, 1):
            q_text = record['question']
            c_text = record['context']
            a_text = record['answer']
            vec = encoder.encode(c_text)

            cur_pg.execute(
                "INSERT INTO rag_entries (question, context, answer, embedding) VALUES (%s, %s, %s, %s);",
                (q_text, c_text, a_text, vec.tolist())
            )

            if idx % 100 == 0:
                print(f"Inserted {idx} records...")
    connection.commit()


def main():
    # Connect to the PostgreSQL instance
    conn_pg = psycopg2.connect(
        host="localhost",
        port=5433,
        dbname="rag_db",
        user="rag_user",
        password="rag_password"
    )

    # Prepare database schema
    initialize_database(conn_pg)

    # Load and trim dataset
    print("Fetching dataset from Hugging Face...")
    hf_data = load_dataset("neural-bridge/rag-dataset-12000", split="test").select(range(500))
    print(f"Loaded {len(hf_data)} entries to ingest.")

    # Set up embedding model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Insert records
    ingest_samples(conn_pg, hf_data, encoder)

    conn_pg.close()
    print("All records have been successfully added to 'rag_entries'.")

if __name__ == "__main__":
    main()
