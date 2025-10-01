// Core data types for the Metta Knowledge Visualizer

export interface Point {
  x: number;
  y: number;
}

// Node and Edge Types
export type NodeType = 'entity' | 'predicate' | 'value' | 'hypergraph';
export type EdgeType = 'relation' | 'hypergraph_connection';

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  position: Point;
  color?: string;
  size?: number;
  isHypergraph?: boolean;
  metadata: {
    originalExpression?: string;
    occurrences?: number; // How many times this node appears
    isGenerated?: boolean; // For hypergraph intermediate nodes
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string; // This is the predicate name (e.g., "gender", "age")
  directed: boolean;
  type: EdgeType;
  color?: string;
  weight?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: GraphMetadata;
  hypergraphs: HypergraphStructure[];
}

export interface GraphMetadata {
  nodeCount: number;
  edgeCount: number;
  hypergraphCount: number;
  lastUpdated: Date;
}

export interface HypergraphStructure {
  id: string;
  predicate: string;
  subjects: string[];
  objects: string[];
  intermediateNodeId?: string; // For visual representation
}

export interface Triple {
  predicate: string;
  subject: string | string[];
  object: string | string[];
  isHypergraph: boolean;
}

export interface ParseError {
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning';
}

export interface ParseResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  errors: ParseError[];
  metadata: GraphMetadata;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ParseError[];
  warnings: ParseError[];
}

// MettaParser Interface
export interface MettaParser {
  parse(mettaText: string): ParseResult;
  validateSyntax(mettaText: string): ValidationResult;
  extractTriples(mettaText: string): Triple[];
  handleHypergraph(expression: string): HypergraphNode[];
}

export interface HypergraphNode {
  id: string;
  predicate: string;
  connections: string[];
  isIntermediate: boolean;
}

export type LayoutAlgorithm = 'force-directed' | 'hierarchical' | 'circular';

export interface LayoutOptions {
  iterations?: number;
  springLength?: number;
  springStrength?: number;
  repulsionStrength?: number;
  damping?: number;
  animationDuration?: number;
  centerForce?: number;
}

export interface LayoutState {
  isAnimating: boolean;
  progress: number;
  algorithm: LayoutAlgorithm;
  startTime: number;
  duration: number;
}

// Component Prop Interfaces
export interface MettaEditorProps {
  initialText: string;
  onTextChange: (text: string) => void;
  onFileUpload: (file: File) => void;
  parseErrors: ParseError[];
}

export interface GraphVisualizerProps {
  graphData: GraphData;
  onNodeSelect: (node: GraphNode) => void;
  onNodeDrag: (nodeId: string, position: Point) => void;
}

export interface UIControlsProps {
  onExportPDF: () => void;
  onExportPNG: () => void;
  showLabels: boolean;
  onToggleLabels: (show: boolean) => void;
  onApplyLayout: (algorithm: LayoutAlgorithm, options?: LayoutOptions) => void;
  layoutState: LayoutState;
  onStopLayout: () => void;
}

export interface LegendProps {
  nodeTypes: Array<{
    type: NodeType;
    color: string;
    label: string;
    count: number;
  }>;
  edgeTypes: Array<{
    type: EdgeType;
    color: string;
    label: string;
    count: number;
  }>;
}

export interface ContextMenuProps {
  position: Point;
  node: GraphNode | null;
  visible: boolean;
  onClose: () => void;
  onDeleteNode: (nodeId: string) => void;
  onEditNode: (nodeId: string) => void;
}