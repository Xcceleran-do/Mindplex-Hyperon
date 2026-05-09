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

// Helper to truncate text to fit width
const fitTextToWidth = (ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string => {
  if (ctx.measureText(text).width <= maxWidth) return text;

  let low = 0;
  let high = text.length;
  let fit = '';

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const str = text.substring(0, mid) + '...';
    if (ctx.measureText(str).width <= maxWidth) {
      fit = str;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return fit || text.substring(0, 3) + '...';
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

  // Get CSS variables for theming
  const computedStyle = getComputedStyle(document.documentElement);
  const nodeFill = computedStyle.getPropertyValue('--node-fill').trim() || '#6b7280';
  const nodeStroke = computedStyle.getPropertyValue('--node-stroke').trim() || 'rgba(0, 0, 0, 0.2)';
  const textPrimary = computedStyle.getPropertyValue('--text-primary').trim() || '#374151';
  const textDimmed = computedStyle.getPropertyValue('--text-dimmed').trim() || 'rgba(55, 65, 81, 0.3)';
  const nodeHighlight = computedStyle.getPropertyValue('--node-highlight').trim() || '#f59e0b';
  const nodeHover = computedStyle.getPropertyValue('--node-hover').trim() || '#0f766e';
  const nodeSelected = computedStyle.getPropertyValue('--node-selected').trim() || '#10b981';

  // Determine colors based on state
  let fillColor = node.color || nodeFill;
  let strokeColor = node.strokeColor || nodeStroke;
  let strokeWidth = 2;
  let currentRadius = radius;

  if (isDimmed) {
    // Dim the node
    fillColor = fillColor.replace(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/, 'rgba($1, $2, $3, 0.2)');
    fillColor = fillColor.replace(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/, 'hsla($1, $2%, $3%, 0.2)');
    fillColor = fillColor.replace(/#([0-9a-f]{6})/i, (match, hex) => {
      const r = parseInt(hex.substr(0, 2), 16);
      const g = parseInt(hex.substr(2, 2), 16);
      const b = parseInt(hex.substr(4, 2), 16);
      return `rgba(${r}, ${g}, ${b}, 0.2)`;
    });
  } else if (isHighlighted) {
    // Highlight the node
    currentRadius *= 1.25;
    strokeColor = nodeHighlight;
    strokeWidth = Math.max(4, 6 * transform.scale);

    // Dynamic glow
    ctx.shadowColor = nodeHighlight;
    ctx.shadowBlur = 15 * transform.scale;
  }

  if (node === hoveredNode) {
    currentRadius *= 1.15;
    strokeWidth = Math.max(3, 4 * transform.scale);
    strokeColor = nodeHover;

    // Subtle hover glow
    ctx.shadowColor = nodeHover;
    ctx.shadowBlur = Math.max(10 * transform.scale, 8);
  }

  if (node === selectedNode) {
    strokeColor = nodeSelected;
    strokeWidth = Math.max(4, 5 * transform.scale);

    // Strong selection glow
    ctx.shadowColor = nodeSelected;
    ctx.shadowBlur = 20 * transform.scale;
  }

  // Draw node based on column type
  if (node.metadata.columnType === 'header') {
    // Calculate the actual font size we'll use for rendering
    const baseFontSize = 14;
    const scaledFontSize = Math.max(baseFontSize, Math.min(baseFontSize * 2, baseFontSize * transform.scale));
    ctx.font = `bold ${scaledFontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;

    // Use width-based truncation for headers (max width to prevent overlap)
    const displayLabel = fitTextToWidth(ctx, node.label, 200);

    // Measure the truncated text with the SAME font we'll use for drawing
    const textMetrics = ctx.measureText(displayLabel);
    const textWidth = textMetrics.width;
    const padding = 24;
    const width = textWidth + padding * 2;
    const height = 36 * transform.scale;

    ctx.fillStyle = fillColor;
    ctx.fillRect(screenPos.x - width / 2, screenPos.y - height / 2, width, height);

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.strokeRect(screenPos.x - width / 2, screenPos.y - height / 2, width, height);

    if (transform.scale > 0.4) {
      // Use explicit text color if available (for headers), otherwise default
      ctx.fillStyle = isDimmed ? textDimmed : (node.textColor || textPrimary);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(displayLabel, screenPos.x, screenPos.y);
    }
  } else {
    // Draw regular nodes as circles
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, currentRadius, 0, 2 * Math.PI);
    ctx.fillStyle = fillColor;
    ctx.fill();

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();

    // Draw node label
    if (transform.scale > 0.4 && node.label) {
      ctx.fillStyle = isDimmed ? textDimmed : (node.textColor || textPrimary);
      const fontSize = 12;
      const scaledFontSize = Math.max(fontSize, Math.min(fontSize * 2, fontSize * transform.scale));
      ctx.font = `${scaledFontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // Smart truncation for value labels (max 150px)
      const displayLabel = fitTextToWidth(ctx, node.label, 140);

      // Position: y + radius + padding
      ctx.fillText(displayLabel, screenPos.x, screenPos.y + currentRadius + 8);
    }
  }

  // Reset shadow
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
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

// Draw column separators and backgrounds
export const drawColumnSeparators = (
  ctx: CanvasRenderingContext2D,
  graphData: GraphData,
  transform: Transform,
  canvasHeight: number
) => {
  const columns = new Map<number, boolean>(); // x -> isTarget

  for (const node of graphData.nodes) {
    if (node.metadata.columnType === 'header') {
      columns.set(node.position.x, !!node.metadata.isTarget);
    }
  }

  // Get separator color from CSS variable
  const computedStyle = getComputedStyle(document.documentElement);
  const separatorColor = computedStyle.getPropertyValue('--separator-color').trim() || 'rgba(0, 0, 0, 0.05)';

  // Draw backgrounds for target columns first
  for (const [columnX, isTarget] of columns) {
    if (isTarget) {
      const screenX = worldToScreen({ x: columnX, y: 0 }, transform).x;
      const width = 250 * transform.scale; // Approximate column width

      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)'; // Subtle highlight
      ctx.fillRect(screenX - width / 2, 0, width, canvasHeight);
    }
  }

  ctx.strokeStyle = separatorColor;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);

  for (const [columnX] of columns) {
    const screenX = worldToScreen({ x: columnX, y: 0 }, transform).x;
    ctx.beginPath();
    ctx.moveTo(screenX, 0);
    ctx.lineTo(screenX, canvasHeight);
    ctx.stroke();
  }

  ctx.setLineDash([]);
};
