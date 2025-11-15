# Project a GDS graph and run GraphSAGE (example; adjust to your GDS version)

// Drop graph if exists
CALL gds.graph.exists('contentGraph_demo') YIELD exists
WITH exists
CALL apoc.do.when(exists,
  'CALL gds.graph.drop("contentGraph_demo") YIELD graphName RETURN graphName',
  'RETURN null' ) YIELD value
RETURN value;

// Project graph
CALL gds.graph.project(
  'contentGraph_demo',
  ['Content','AudienceSegment','Topic','Creator'],
  {
    ENGAGED_WITH: {properties:['views','likes','watch_time']},
    HAS_TOPIC: {},
    CREATED: {}
  }
)
YIELD graphName, nodeCount, relationshipCount;

// Example GraphSAGE write - API may differ by GDS version
CALL gds.beta.graphSage.write('contentGraph_demo', {
  writeProperty: 'gsage_embedding_v1',
  featureProperties: ['lengthSec'],
  embeddingDimension: 128,
  epochs: 10,
  learningRate: 0.01,
  numSamples: [25,10],
  aggregator: 'mean'
}) YIELD nodes, embeddingDimension;

RETURN nodes, embeddingDimension;
