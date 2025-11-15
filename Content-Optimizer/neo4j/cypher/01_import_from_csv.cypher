// Import content and creators from CSV located in Neo4j import directory (file:///sample_content.csv)
// This file is idempotent and uses MERGE so it can be re-run.

// Import from CSV with idempotent MERGE upserts
LOAD CSV WITH HEADERS FROM 'file:///sample_content.csv' AS row
FIELDTERMINATOR ','
MERGE (c:Content {contentId: row.contentId})
SET c.title = row.title,
    c.lengthSec = toInteger(row.lengthSec),
    c.format = row.format,
    c.platform = row.platform,
    c.createdAt = coalesce(c.createdAt, datetime(row.createdAt))
MERGE (u:Creator {creatorId: row.creatorId})
SET u.createdAt = coalesce(u.createdAt, datetime())
MERGE (u)-[:CREATED]->(c)
RETURN 'import_completed' AS status;
