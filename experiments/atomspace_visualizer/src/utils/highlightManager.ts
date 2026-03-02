// Highlight state management utilities for ColumnarVisualizer
import { GraphData, FilterState, HighlightState } from '../types';

// Update highlight state based on selected node and filter
export const updateHighlightState = (
  graphData: GraphData,
  filterState: FilterState
): HighlightState => {
  const highlighted = new Set<string>();
  const highlightedEdges = new Set<string>();
  const dimmed = new Set<string>();
  const dimmedEdges = new Set<string>();

  if (filterState.active) {
    // AND logic for property filters and articles
    let filteredNodeIds = new Set(graphData.nodes.map(n => n.id));
    let filteredEdgeIds = new Set(graphData.edges.map(e => e.id));

    // Multi-article selection (AND): only nodes/edges connected to ALL selected articles
    if (filterState.articleIds && filterState.articleIds.length > 0) {
      for (const articleId of filterState.articleIds) {
        const articleNodeId = `article-${articleId}`;
        // Only keep nodes/edges connected to this article
        const connectedNodeIds = new Set<string>();
        const connectedEdgeIds = new Set<string>();
        for (const edge of graphData.edges) {
          if (edge.source === articleNodeId) {
            connectedEdgeIds.add(edge.id);
            connectedNodeIds.add(edge.target);
          }
        }
        filteredNodeIds = new Set([...filteredNodeIds].filter(id => connectedNodeIds.has(id) || id === articleNodeId));
        filteredEdgeIds = new Set([...filteredEdgeIds].filter(id => connectedEdgeIds.has(id)));
      }
    }

    // Property filters: OR within property, AND across properties
    if (filterState.propertyFilters && filterState.propertyFilters.length > 0) {
      // Group filters by property
      const propertyGroups: Record<string, string[]> = {};
      for (const filter of filterState.propertyFilters) {
        if (!propertyGroups[filter.property]) propertyGroups[filter.property] = [];
        propertyGroups[filter.property].push(filter.value);
      }
      // For each property, get all nodes/edges matching any value (OR)
      let propertyNodeSets: Array<Set<string>> = [];
      let propertyEdgeSets: Array<Set<string>> = [];
      for (const property in propertyGroups) {
        const values = propertyGroups[property];
        const nodeSet = new Set<string>();
        const edgeSet = new Set<string>();
        for (const value of values) {
          const propertyNodeId = `${property}-${value}`;
          for (const edge of graphData.edges) {
            if (edge.target === propertyNodeId) {
              edgeSet.add(edge.id);
              nodeSet.add(edge.source);
            }
          }
          nodeSet.add(propertyNodeId);
        }
        propertyNodeSets.push(nodeSet);
        propertyEdgeSets.push(edgeSet);
      }
      // AND across properties: intersection of all property sets
      if (propertyNodeSets.length > 0) {
        let intersectionNodes = propertyNodeSets[0];
        for (let i = 1; i < propertyNodeSets.length; i++) {
          intersectionNodes = new Set([...intersectionNodes].filter(x => propertyNodeSets[i].has(x)));
        }
        filteredNodeIds = new Set([...filteredNodeIds].filter(id => intersectionNodes.has(id)));
      }
      if (propertyEdgeSets.length > 0) {
        let intersectionEdges = propertyEdgeSets[0];
        for (let i = 1; i < propertyEdgeSets.length; i++) {
          intersectionEdges = new Set([...intersectionEdges].filter(x => propertyEdgeSets[i].has(x)));
        }
        filteredEdgeIds = new Set([...filteredEdgeIds].filter(id => intersectionEdges.has(id)));
      }
    }

    // Legacy single property filter (AND)
    if (filterState.property && filterState.value) {
      const propertyNodeId = `${filterState.property}-${filterState.value}`;
      const connectedNodeIds = new Set<string>();
      const connectedEdgeIds = new Set<string>();
      for (const edge of graphData.edges) {
        if (edge.target === propertyNodeId) {
          connectedEdgeIds.add(edge.id);
          connectedNodeIds.add(edge.source);
        }
      }
      filteredNodeIds = new Set([...filteredNodeIds].filter(id => connectedNodeIds.has(id) || id === propertyNodeId));
      filteredEdgeIds = new Set([...filteredEdgeIds].filter(id => connectedEdgeIds.has(id)));
    }

    // Highlight and dim
    for (const id of filteredNodeIds) highlighted.add(id);
    for (const id of filteredEdgeIds) highlightedEdges.add(id);
    for (const node of graphData.nodes) {
      if (!highlighted.has(node.id)) dimmed.add(node.id);
    }
    for (const edge of graphData.edges) {
      if (!highlightedEdges.has(edge.id)) dimmedEdges.add(edge.id);
    }
  }

  return {
    highlightedNodes: highlighted,
    highlightedEdges: highlightedEdges,
    dimmedNodes: dimmed,
    dimmedEdges: dimmedEdges
  };
};
