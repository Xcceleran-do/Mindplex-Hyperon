// Legend component for displaying node and edge types
import { Component, createEffect, createSignal } from 'solid-js';
import { GraphData } from '../../types';
import { getNodeColor, getEdgeColor } from '../../utils/nodeUtils';

export interface LegendProps {
  graphData: GraphData;
}

const Legend: Component<LegendProps> = (props) => {
  const [nodeLabels, setNodeLabels] = createSignal<Set<string>>(new Set());
  const [predicateTypes, setPredicateTypes] = createSignal<Set<string>>(new Set());

  // Extract unique node labels and predicate types from graph data
  createEffect(() => {
    const nodes = props.graphData.nodes;
    const edges = props.graphData.edges;

    const uniqueNodeLabels = new Set<string>();
    const uniquePredicateTypes = new Set<string>();

    // Collect unique node labels
    nodes.forEach(node => {
      if (node.label) uniqueNodeLabels.add(node.label);
    });

    // Collect unique predicate names (edge labels are the predicates)
    edges.forEach(edge => {
      if (edge.label) uniquePredicateTypes.add(edge.label);
    });

    setNodeLabels(uniqueNodeLabels);
    setPredicateTypes(uniquePredicateTypes);
  });

  return (
    <>
      {nodeLabels().size > 0 && (
        <div class="legend-section">
          <h4>Nodes</h4>
          {Array.from(nodeLabels()).sort().map(label => (
            <div class="legend-item">
              <div 
                class="legend-color" 
                style={`background-color: ${getNodeColor(label)}`}
              ></div>
              <span>{label}</span>
            </div>
          ))}
        </div>
      )}

      {predicateTypes().size > 0 && (
        <div class="legend-section">
          <h4>Predicates</h4>
          {Array.from(predicateTypes()).sort().map(predicate => (
            <div class="legend-item">
              <div 
                class="legend-color" 
                style={`background-color: ${getEdgeColor(predicate)}`}
              ></div>
              <span>{predicate}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
};

export default Legend;