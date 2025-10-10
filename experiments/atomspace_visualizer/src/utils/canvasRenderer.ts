// Canvas rendering utilities for ColumnarVisualizer
import { GraphNode, GraphEdge, GraphData, Point, HighlightState } from '../types';

export interface Transform {
  x: number;
  y: number;
  scale: number;
}

// Transform screen coordinates to world coordinates
export const screenToWorld = (screenPoint: Point, transform: Transform): Point => {
  return {
    x: (screenPoint.x - transform.x) / transform.scale,
    y: (screenPoint.y - transform.y) / transform.scale
  };
};

// Transform world coordinates to screen coordinates
export const worldToScreen = (worldPoint: Point, transform: Transform): Point => {
  return {
    x: worldPoint.x * transform.scale + transform.x,
    y: worldPoint.y * transform.scale + transform.y
  };
};

// Render a single node with columnar styling
export const renderNode = (
  ctx: CanvasRenderingContext2D,
  node: GraphNode,
  transform: Transform,
  highlightState: HighlightState,
  hoveredNode: GraphNode | null,
  selectedNode: GraphNode | null,
  canvasWidth: number,
  canvasHeight: number
) => {
  const screenPos = worldToScreen(node.position, transform);
  const baseSize = node.size || 50;
  const radius = (baseSize / 2) * transform.scale;

  // Skip rendering if node is outside viewport
  const margin = radius + 50;
  if (screenPos.x < -margin || screenPos.x > canvasWidth + margin ||
      screenPos.y < -margin || screenPos.y > canvasHeight + margin) {
    return;
  }

  const isHighlighted = highlightState.highlightedNodes.has(node.id);
  const isDimmed = highlightState.dimmedNodes.has(node.id);

  // Determine colors based on state
  let fillColor = node.color || '#6b7280';
  let strokeColor = 'rgba(0, 0, 0, 0.2)';
  let strokeWidth = 2;
  let currentRadius = radius;

  if (isDimmed) {
    // Dim the node
    fillColor = fillColor.replace(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/, 'rgba($1, $2, $3, 0.2)');
    fillColor = fillColor.replace(/#([0-9a-f]{6})/i, (match, hex) => {
      const r = parseInt(hex.substr(0, 2), 16);
      const g = parseInt(hex.substr(2, 2), 16);
      const b = parseInt(hex.substr(4, 2), 16);
      return `rgba(${r}, ${g}, ${b}, 0.2)`;
    });
  } else if (isHighlighted) {
    // Highlight the node
    currentRadius *= 1.2;
    strokeColor = '#f59e0b';
    strokeWidth = 4;
    
    // Add glow effect
    ctx.shadowColor = '#f59e0b';
    ctx.shadowBlur = 20;
  }

  if (node === hoveredNode) {
    currentRadius *= 1.1;
    strokeWidth = 3;
    strokeColor = '#3b82f6';
  }

  if (node === selectedNode) {
    strokeColor = '#10b981';
    strokeWidth = 4;
  }

  // Draw node based on column type
  if (node.metadata.columnType === 'header') {
    // Draw header as rectangle with flexible width based on text
    const fontSize = 14;
    ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
    const textMetrics = ctx.measureText(node.label);
    const textWidth = textMetrics.width;
    const padding = 40; // Padding on each side (increased for longer names)
    const width = (textWidth + padding * 2) * transform.scale;
    const height = 40 * transform.scale;
    
    ctx.fillStyle = fillColor;
    ctx.fillRect(screenPos.x - width / 2, screenPos.y - height / 2, width, height);
    
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.strokeRect(screenPos.x - width / 2, screenPos.y - height / 2, width, height);
  } else {
    // Draw regular nodes as circles
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, currentRadius, 0, 2 * Math.PI);
    ctx.fillStyle = fillColor;
    ctx.fill();
    
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
  }

  // Reset shadow
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;

  // Draw node label
  if (transform.scale > 0.4 && node.label) {
    ctx.fillStyle = isDimmed ? 'rgba(55, 65, 81, 0.3)' : '#374151';
    const fontSize = node.metadata.columnType === 'header' ? 14 : 12;
    ctx.font = `${Math.max(fontSize, fontSize * transform.scale)}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Truncate label if too long
    let displayLabel = node.label;
    if (displayLabel.length > 20) {
      displayLabel = displayLabel.substring(0, 18) + '...';
    }
    
    ctx.fillText(displayLabel, screenPos.x, screenPos.y);
  }
};

// Get unique color for each article's line
export const getArticleLineColor = (articleId: string, isHighlighted: boolean, isDimmed: boolean): string => {
  if (isDimmed) return 'rgba(107, 114, 128, 0.1)';
  if (isHighlighted) return '#f59e0b';
  
  // Generate unique color based on article ID
  const match = articleId.match(/article-(\d+)/);
  if (match) {
    const articleNum = parseInt(match[1]);
    const hue = (articleNum * 137) % 360; // Golden angle for good color distribution
    return `hsla(${hue}, 70%, 55%, 0.7)`;
  }
  
  return 'rgba(107, 114, 128, 0.4)';
};

// Render edges as continuous curved lines from article through properties
export const renderArticleConnections = (
  ctx: CanvasRenderingContext2D,
  articleId: string,
  graphData: GraphData,
  transform: Transform,
  highlightState: HighlightState
) => {
  const articleNode = graphData.nodes.find(n => n.id === articleId);
  if (!articleNode) return;

  // Find all edges from this article, sorted by column position
  const edges = graphData.edges
    .filter(e => e.source === articleId)
    .sort((a, b) => {
      const nodeA = graphData.nodes.find(n => n.id === a.target);
      const nodeB = graphData.nodes.find(n => n.id === b.target);
      return (nodeA?.position.x || 0) - (nodeB?.position.x || 0);
    });

  if (edges.length === 0) return;

  const isHighlighted = highlightState.highlightedNodes.has(articleId);
  const isDimmed = highlightState.dimmedNodes.has(articleId);

  // Determine styling with unique color per article
  const strokeColor = getArticleLineColor(articleId, isHighlighted, isDimmed);
  let lineWidth = 2.5;

  if (isDimmed) {
    lineWidth = 1;
  } else if (isHighlighted) {
    lineWidth = 4;
  }

  // Draw continuous curved line through all property values
  ctx.beginPath();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineWidth * transform.scale;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const articleScreen = worldToScreen(articleNode.position, transform);
  ctx.moveTo(articleScreen.x, articleScreen.y);

  // Draw smooth curve through each property node
  for (let i = 0; i < edges.length; i++) {
    const targetNode = graphData.nodes.find(n => n.id === edges[i].target);
    if (!targetNode) continue;

    const targetScreen = worldToScreen(targetNode.position, transform);

    if (i === 0) {
      // First segment: quadratic curve from article to first property
      const controlX = (articleScreen.x + targetScreen.x) / 2;
      const controlY = articleScreen.y;
      ctx.quadraticCurveTo(controlX, controlY, targetScreen.x, targetScreen.y);
    } else {
      // Subsequent segments: smooth curve to next property
      const prevNode = graphData.nodes.find(n => n.id === edges[i - 1].target);
      if (!prevNode) continue;
      const prevScreen = worldToScreen(prevNode.position, transform);
      
      const controlX = (prevScreen.x + targetScreen.x) / 2;
      const controlY = (prevScreen.y + targetScreen.y) / 2;
      ctx.quadraticCurveTo(controlX, controlY, targetScreen.x, targetScreen.y);
    }
  }

  ctx.stroke();

  // Draw connection markers at each node for better line tracking
  if (!isDimmed && transform.scale > 0.5) {
    ctx.fillStyle = strokeColor;
    // Marker at article start
    ctx.beginPath();
    ctx.arc(articleScreen.x, articleScreen.y, 5 * transform.scale, 0, 2 * Math.PI);
    ctx.fill();
    
    // Markers at property connections
    for (const edge of edges) {
      const targetNode = graphData.nodes.find(n => n.id === edge.target);
      if (!targetNode) continue;
      const targetScreen = worldToScreen(targetNode.position, transform);
      ctx.beginPath();
      ctx.arc(targetScreen.x, targetScreen.y, 5 * transform.scale, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
};

// Draw column separators
export const drawColumnSeparators = (
  ctx: CanvasRenderingContext2D,
  graphData: GraphData,
  transform: Transform,
  canvasHeight: number
) => {
  const columns = new Set<number>();
  for (const node of graphData.nodes) {
    if (node.metadata.columnType === 'header') {
      columns.add(node.position.x);
    }
  }

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);

  for (const columnX of columns) {
    const screenX = worldToScreen({ x: columnX, y: 0 }, transform).x;
    ctx.beginPath();
    ctx.moveTo(screenX, 0);
    ctx.lineTo(screenX, canvasHeight);
    ctx.stroke();
  }

  ctx.setLineDash([]);
};
