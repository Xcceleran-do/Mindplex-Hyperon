# db.py
# Neo4j database operations for CCRS
# (FR-1.1, FR-1.2)

import logging
from neo4j import GraphDatabase, basic_auth
import pandas as pd
import os
from dotenv import load_dotenv  # Add this import

# Configure logging (FR-5.2)
logging.basicConfig(
    filename='ccrs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Neo4j connection details (now loaded from .env)
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USER')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
)

def setup_schema():
    """
    Create unique constraints and indexes in Neo4j (FR-1.1, Appendix 6.2)
    """
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Content) REQUIRE c.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)")
        logging.info("Neo4j schema setup completed.")
    except Exception as e:
        logging.error(f"Schema setup failed: {e}")
        raise

def create_content(tx, row):
    """
    Create Content node and compute engagement_score (FR-1.1)
    """
    engagement_score = (
        row['likes'] + row['shares'] * 1.5 + row['comments'] * 2 + row['views'] * 0.05
    )
    tx.run(
        """
        MERGE (c:Content {id: $id})
        SET c.text = $text, c.length = $length, c.engagement_score = $engagement_score
        """,
        id=str(row['post_id']),
        text=row['text'],
        length=int(row['length']),
        engagement_score=float(engagement_score)
    )

def link_topic(tx, row):
    tx.run(
        """
        MERGE (c:Content {id: $id})
        MERGE (t:Topic {name: $topic})
        MERGE (c)-[:HAS_TOPIC]->(t)
        """,
        id=str(row['post_id']),
        topic=row['topic']
    )

def link_style(tx, row):
    tx.run(
        """
        MERGE (c:Content {id: $id})
        MERGE (s:Style {name: $style})
        MERGE (c)-[:HAS_STYLE]->(s)
        """,
        id=str(row['post_id']),
        style=row['style']
    )

def link_structure(tx, row):
    tx.run(
        """
        MERGE (c:Content {id: $id})
        MERGE (st:Structure {name: $structure})
        MERGE (c)-[:HAS_STRUCTURE]->(st)
        """,
        id=str(row['post_id']),
        structure=row['structure']
    )

def import_csv(csv_path):
    """
    Import content data from CSV and create nodes/relationships (FR-1.1)
    """
    required_cols = [
        'post_id', 'text', 'length', 'topic', 'style', 'structure', 'likes', 'shares', 'comments', 'views'
    ]
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Found: {df.columns}")
        count = 0
        with driver.session() as session:
            for _, row in df.iterrows():
                try:
                    session.write_transaction(create_content, row)
                    session.write_transaction(link_topic, row)
                    session.write_transaction(link_style, row)
                    session.write_transaction(link_structure, row)
                    count += 1
                except Exception as e:
                    logging.error(f"Error importing row {row['post_id']}: {e}")
        logging.info(f"{count} posts imported successfully from {csv_path}.")
        print(f"{count} posts imported successfully.")
    except Exception as e:
        logging.error(f"CSV import failed: {e}")
        print(f"CSV import failed: {e}")

def fetch_data(topic_filter=None):
    """
    Fetch all Content nodes and their relationships (FR-1.2)
    Returns: list of dicts
    """
    query = (
        """
        MATCH (c:Content)-[:HAS_TOPIC]->(t:Topic),
              (c)-[:HAS_STYLE]->(s:Style),
              (c)-[:HAS_STRUCTURE]->(st:Structure)
        """
    )
    if topic_filter:
        query += " WHERE t.name = $topic "
    query += " RETURN c, t, s, st "
    try:
        with driver.session() as session:
            result = session.run(query, topic=topic_filter) if topic_filter else session.run(query)
            data = []
            for record in result:
                c = record['c']
                t = record['t']
                s = record['s']
                st = record['st']
                data.append({
                    'id': c['id'],
                    'text': c['text'],
                    'length': c['length'],
                    'engagement_score': c['engagement_score'],
                    'topic': t['name'],
                    'style': s['name'],
                    'structure': st['name']
                })
            return data
    except Exception as e:
        logging.error(f"Data fetch failed: {e}")
        return []

# Optional: Close driver on exit
def close():
    driver.close()

if __name__ == "__main__":
    # Example usage for manual testing
    setup_schema()
    import_csv(os.path.join('data', 'sample_data.csv'))
    # Print a sample of fetched data
    data = fetch_data()
    print(f"Fetched {len(data)} records.")

