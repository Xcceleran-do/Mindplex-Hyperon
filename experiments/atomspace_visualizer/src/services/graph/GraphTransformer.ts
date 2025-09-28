// Graph data transformation service
// Converts parsed Metta expressions to graph data structures
import { 
  Triple, 
  GraphNode, 
  GraphEdge, 
  HypergraphStructure, 
  GraphData, 
  GraphMetadata,
  Point 
} from '../../types';
import { createOrUpdateNode, getNodeColor, getEdgeColor } from '../../utils/nodeUtils';

export interface GraphTransformer {
  transformTriplestoGraph(triples: Triple[]): GraphData;
  processSimpleTriple(triple: Triple, nodes: GraphNode[], edges: GraphEdge[], index: number): void;
  processHypergraphTriple(triple: Triple, nodes: GraphNode[], edges: GraphEdge[], hypergraphs: HypergraphStructure[], index: number): void;
  detectBidirectionalRelationships(edges: GraphEdge[]): GraphEdge[];
  createIntermediateNode(predicate: string, counter: number, position: Point): GraphNode;
}

export class GraphTransformerImpl implements GraphTransformer {
  private hypergraphCounter = 0;
  private bidirectionalEdges = new Set<string>();

  transformTriplestoGraph(triples: Triple[]): GraphData {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const hypergraphs: HypergraphStructure[] = [];

    // Process each triple to create nodes and edges
    for (let index = 0; index < triples.length; index++) {
      const triple = triples[index];
      
      if (triple.isHypergraph) {
        // Handle hypergraph structures
        const hypergraphStructures = this.createHypergraphStructures(triple);
        hypergraphs.push(...hypergraphStructures);
        this.processHypergraphTriple(triple, nodes, edges, hypergraphs, index);
      } else {
        // Handle simple triplets
        this.processSimpleTriple(triple, nodes, edges, index);
      }
    }

    // Detect and handle bidirectional relationships
    const processedEdges = this.detectBidirectionalRelationships(edges);

    return {
      nodes,
      edges: processedEdges,
      hypergraphs,
      metadata: this.createGraphMetadata(nodes, processedEdges, hypergraphs)
    };
  }

  processSimpleTriple(triple: Triple, nodes: GraphNode[], edges: GraphEdge[], index: number): void {
    const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
    const objects = Array.isArray(triple.object) ? triple.object : [triple.object];

    // Create nodes for subjects and objects with proper positioning
    const subjectNodes = subjects.map((subject, i) =>
      createOrUpdateNode(nodes, subject, { 
        x: i * 120 - (subjects.length - 1) * 60, 
        y: index * 100 
      })
    );

    const objectNodes = objects.map((object, i) =>
      createOrUpdateNode(nodes, object, { 
        x: (subjects.length + i) * 120 - (subjects.length - 1) * 60, 
        y: index * 100 
      })
    );

    // Set node types and colors
    this.setNodeProperties([...subjectNodes, ...objectNodes], triple.predicate);

    // Create edges from each subject to each object
    this.createEdgesBetweenNodes(subjectNodes, objectNodes, triple, edges);
  }

  processHypergraphTriple(
    triple: Triple, 
    nodes: GraphNode[], 
    edges: GraphEdge[], 
    hypergraphs: HypergraphStructure[], 
    index: number
  ): void {
    const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
    const objects = Array.isArray(triple.object) ? triple.object : [triple.object];

    // Create nodes for all entities in a circular arrangement around the center
    const allEntities = [...subjects, ...objects];
    const centerX = 0;
    const centerY = index * 150;
    const radius = Math.max(80, allEntities.length * 20);

    const entityNodes = allEntities.map((entity, i) => {
      const angle = (2 * Math.PI * i) / allEntities.length;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      return createOrUpdateNode(nodes, entity, { x, y });
    });

    // Create intermediate node for hypergraph at the center
    const hypergraph = hypergraphs[hypergraphs.length - 1]; // Most recent hypergraph
    if (hypergraph && hypergraph.intermediateNodeId) {
      const intermediateNode = this.createIntermediateNode(
        triple.predicate, 
        this.hypergraphCounter, 
        { x: centerX, y: centerY }
      );
      
      // Add the intermediate node to the nodes array
      nodes.push(intermediateNode);

      // Connect all entities to the intermediate node
      this.createHypergraphConnections(entityNodes, intermediateNode, triple, edges);
    }

    // Set colors for entity nodes
    this.setNodeProperties(entityNodes, triple.predicate);
  }

  detectBidirectionalRelationships(edges: GraphEdge[]): GraphEdge[] {
    const edgeMap = new Map<string, GraphEdge>();
    const bidirectionalPairs = new Set<string>();
    const processedEdges: GraphEdge[] = [];

    // First pass: identify potential bidirectional relationships
    for (const edge of edges) {
      const forwardKey = `${edge.source}-${edge.target}-${edge.label}`;
      const reverseKey = `${edge.target}-${edge.source}-${edge.label}`;
      
      edgeMap.set(forwardKey, edge);
      
      // Check if reverse edge exists
      if (edgeMap.has(reverseKey)) {
        const pairKey = [forwardKey, reverseKey].sort().join('|');
        bidirectionalPairs.add(pairKey);
      }
    }

    // Second pass: create processed edges
    const processedEdgeIds = new Set<string>();
    
    for (const edge of edges) {
      const forwardKey = `${edge.source}-${edge.target}-${edge.label}`;
      const reverseKey = `${edge.target}-${edge.source}-${edge.label}`;
      const pairKey = [forwardKey, reverseKey].sort().join('|');
      
      // Always keep directed edges
      if (!processedEdgeIds.has(forwardKey)) {
        processedEdges.push(edge);
        processedEdgeIds.add(forwardKey);
      }
    }

    return processedEdges;
  }

  createIntermediateNode(predicate: string, counter: number, position: Point): GraphNode {
    const intermediateNodeId = `${predicate}-group-${counter}`;
    
    return {
      id: intermediateNodeId,
      label: `${predicate} group`,
      type: 'hypergraph',
      position,
      isHypergraph: true,
      color: getNodeColor(predicate),
      size: 1.2, // Slightly larger for hypergraph nodes
      metadata: {
        isGenerated: true,
        originalExpression: `${predicate} hypergraph`,
        occurrences: 1
      }
    };
  }

  private createHypergraphStructures(triple: Triple): HypergraphStructure[] {
    const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
    const objects = Array.isArray(triple.object) ? triple.object : [triple.object];
    
    this.hypergraphCounter++;
    
    return [{
      id: `hypergraph-${this.hypergraphCounter}`,
      predicate: triple.predicate,
      subjects,
      objects,
      intermediateNodeId: `${triple.predicate}-group-${this.hypergraphCounter}`
    }];
  }

  private setNodeProperties(nodes: GraphNode[], predicate: string): void {
    nodes.forEach(node => {
      if (!node.color) {
        node.color = getNodeColor(node.label);
      }
      
      // Determine node type based on context
      if (!node.type || node.type === 'entity') {
        node.type = this.determineNodeType(node.label, predicate);
      }
    });
  }

  private determineNodeType(label: string, predicate: string): 'entity' | 'predicate' | 'value' | 'hypergraph' {
    // Simple heuristics to determine node type
    if (label.match(/^[A-Z][a-z]+$/)) {
      return 'entity'; // Proper names like "John", "Mary"
    }
    
    if (label.match(/^[0-9]+$/) || label.match(/^[MF]$/) || label.match(/^(true|false)$/i)) {
      return 'value'; // Numbers, gender markers, booleans
    }
    
    if (predicate === label) {
      return 'predicate';
    }
    
    return 'entity'; // Default
  }

  private createEdgesBetweenNodes(
    sourceNodes: GraphNode[], 
    targetNodes: GraphNode[], 
    triple: Triple, 
    edges: GraphEdge[]
  ): void {
    sourceNodes.forEach(sourceNode => {
      targetNodes.forEach(targetNode => {
        const edgeId = `${triple.predicate}-${sourceNode.id}-${targetNode.id}`;
        
        // Avoid duplicate edges
        if (!this.edgeExists(edges, edgeId)) {
          edges.push({
            id: edgeId,
            source: sourceNode.id,
            target: targetNode.id,
            label: triple.predicate,
            directed: true,
            type: 'relation',
            color: getEdgeColor(triple.predicate)
          });
        }
      });
    });
  }

  private createHypergraphConnections(
    entityNodes: GraphNode[], 
    intermediateNode: GraphNode, 
    triple: Triple, 
    edges: GraphEdge[]
  ): void {
    entityNodes.forEach(entityNode => {
      const edgeId = `${triple.predicate}-${entityNode.id}-${intermediateNode.id}`;
      
      if (!this.edgeExists(edges, edgeId)) {
        edges.push({
          id: edgeId,
          source: entityNode.id,
          target: intermediateNode.id,
          label: triple.predicate,
          directed: false, // Hypergraph connections are typically undirected
          type: 'hypergraph_connection',
          color: getEdgeColor(triple.predicate)
        });
      }
    });
  }

  private edgeExists(edges: GraphEdge[], edgeId: string): boolean {
    return edges.some(edge => edge.id === edgeId);
  }

  private createGraphMetadata(
    nodes: GraphNode[], 
    edges: GraphEdge[], 
    hypergraphs: HypergraphStructure[]
  ): GraphMetadata {
    return {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      hypergraphCount: hypergraphs.length,
      lastUpdated: new Date()
    };
  }
}