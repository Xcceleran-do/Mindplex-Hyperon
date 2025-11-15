// STEP 3: Graph projection + GraphSAGE embeddings
// This script:
// 1. Computes an engagementScore on Content nodes from ENGAGED_WITH edges.
// 2. Drops prior GDS graph if it exists.
// 3. Projects a named graph 'contentGraph' with selected labels & relationships.
// 4. Trains a GraphSAGE model using Content node features (lengthSec, engagementScore).
// 5. Writes embeddings back to the database under property 'embedding'.
// 6. Shows a preview of embeddings.
// Safe/idempotent: can be re-run; graph is dropped and recreated.

// 1. Compute engagementScore feature (sum of views + likes per Content)
MATCH (c:Content)
OPTIONAL MATCH (s:AudienceSegment)-[e:ENGAGED_WITH]->(c)
WITH c, coalesce(sum(e.views),0) AS v, coalesce(sum(e.likes),0) AS l
SET c.engagementScore = v + l;

// Ensure all nodes have numeric feature placeholders to avoid NaN errors
MATCH (n)
WHERE n.lengthSec IS NULL
SET n.lengthSec = 0;

MATCH (n)
WHERE n.engagementScore IS NULL
SET n.engagementScore = 0;

// Drop model if it already exists to allow reruns without manual cleanup
CALL {
  WITH 'contentGraphSageLatest' AS targetName
  CALL gds.model.list() YIELD modelName
  WITH collect(modelName) AS existing, targetName
  WHERE targetName IN existing
  CALL gds.model.drop(targetName) YIELD modelName
  RETURN modelName
} RETURN 1;

// 2. Drop existing projected graph if present (ignore if missing)
CALL gds.graph.drop('contentGraph', false) YIELD graphName;

// 3. Project new graph with node properties required for embeddings
CALL gds.graph.project(
  'contentGraph',
  ['Content','Creator','Topic','AudienceSegment'],
  {
    CREATED: {orientation: 'UNDIRECTED'},
    HAS_TOPIC: {orientation: 'UNDIRECTED'},
  ENGAGED_WITH: {orientation: 'UNDIRECTED'}
  },
  {nodeProperties: ['lengthSec','engagementScore']}
) YIELD graphName, nodeCount, relationshipCount;

// 4. Train GraphSAGE model (using beta API; adjust if non-beta in your GDS version)

CALL gds.beta.graphSage.train('contentGraph', {
  modelName: 'contentGraphSageLatest',
  featureProperties: ['lengthSec','engagementScore'],
  embeddingDimension: 64,
  epochs: 4,
  batchSize: 64,
  sampleSizes: [4,4],
  aggregator: 'mean'
}) YIELD modelInfo, trainMillis;

// 5. Write embeddings back to DB
CALL gds.beta.graphSage.write('contentGraph', {
  modelName: 'contentGraphSageLatest',
  writeProperty: 'embedding'
}) YIELD nodePropertiesWritten, computeMillis;

// 6. Preview embeddings for a few Content nodes
MATCH (c:Content)
RETURN c.contentId AS contentId, size(c.embedding) AS dim, c.embedding[0..5] AS firstValues, c.engagementScore AS engagementScore
ORDER BY contentId
LIMIT 5;

RETURN 'gsage_step3_completed' AS status;
