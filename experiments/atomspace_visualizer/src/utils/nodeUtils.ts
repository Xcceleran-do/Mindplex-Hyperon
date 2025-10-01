// Utility functions for node management
import { GraphNode } from '../types';

/**
 * Generate a consistent node ID based on the label
 * Nodes with the same label will get the same ID
 */
export function generateNodeId(label: string): string {
  // Convert to lowercase and replace spaces/special chars with hyphens
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Create or update a node in the nodes array
 * If a node with the same label exists, increment its occurrence count
 */
export function createOrUpdateNode(
  nodes: GraphNode[], 
  label: string, 
  position?: { x: number; y: number }
): GraphNode {
  const nodeId = generateNodeId(label);
  
  // Check if node already exists
  let existingNode: GraphNode | undefined;
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].id === nodeId) {
      existingNode = nodes[i];
      break;
    }
  }
  
  if (existingNode) {
    // Update occurrence count
    existingNode.metadata.occurrences = (existingNode.metadata.occurrences || 0) + 1;
    return existingNode;
  } else {
    // Create new node
    const newNode: GraphNode = {
      id: nodeId,
      label: label,
      type: 'entity', // Default type, can be refined later
      position: position || { x: 0, y: 0 },
      metadata: {
        occurrences: 1
      }
    };
    nodes.push(newNode);
    return newNode;
  }
}

/**
 * Parse a simple Metta expression like "(gender Chandler M)"
 * Returns the predicate and arguments
 */
export function parseMettaExpression(expression: string): {
  predicate: string;
  args: string[];
} | null {
  // Remove outer parentheses and split by spaces
  const trimmed = expression.trim();
  if (trimmed.charAt(0) !== '(' || trimmed.charAt(trimmed.length - 1) !== ')') {
    return null;
  }
  
  const content = trimmed.slice(1, -1).trim();
  const parts = content.split(/\s+/);
  
  if (parts.length < 2) {
    return null;
  }
  
  return {
    predicate: parts[0],
    args: parts.slice(1)
  };
}

/**
 * Generate a color for a node based on its label
 */
export function getNodeColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsla(${hue}, 65%, 70%, 0.7)`;
}

/**
 * Generate a color for an edge based on its predicate
 */
export function getEdgeColor(predicate: string): string {
  let hash = 0;
  for (let i = 0; i < predicate.length; i++) {
    hash = predicate.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = (Math.abs(hash) + 180) % 360; // Offset to differentiate from nodes
  return `hsla(${hue}, 60%, 65%, 0.6)`;
}