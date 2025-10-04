// Columnar Visualizer component - property-based columnar layout
import { Component, onMount, createEffect, onCleanup, createSignal } from 'solid-js';
import { GraphData, GraphNode, GraphEdge, Point, FilterState, HighlightState } from '../../types';
import styles from './ColumnarVisualizer.module.css';

export interface ColumnarVisualizerProps {
  graphData: GraphData;
  onNodeSelect: (node: GraphNode) => void;
  filterState: FilterState;
  onFilterChange: (filter: FilterState) => void;
}

interface Transform {
  x: number;
  y: number;
  scale: number;
}

const ColumnarVisualizer: Component<ColumnarVisualizerProps> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let animationFrameId: number;
  
  // Canvas transformation state
  let transform: Transform = { x: 50, y: 50, scale: 0.65 };
  let isPanning = false;
  let lastPanPoint: Point = { x: 0, y: 0 };
  let hoveredNode: GraphNode | null = null;
  let selectedNode: GraphNode | null = null;

  // Highlighting state
  const [highlightState, setHighlightState] = createSignal<HighlightState>({
    highlightedNodes: new Set(),
    highlightedEdges: new Set(),
    dimmedNodes: new Set(),
    dimmedEdges: new Set()
  });

  // Transform screen coordinates to world coordinates
  const screenToWorld = (screenPoint: Point): Point => {
    return {
      x: (screenPoint.x - transform.x) / transform.scale,
      y: (screenPoint.y - transform.y) / transform.scale
    };
  };

  // Transform world coordinates to screen coordinates
  const worldToScreen = (worldPoint: Point): Point => {
    return {
      x: worldPoint.x * transform.scale + transform.x,
      y: worldPoint.y * transform.scale + transform.y
    };
  };

  // Get mouse position relative to canvas
  const getMousePos = (e: MouseEvent): Point => {
    if (!canvasRef) return { x: 0, y: 0 };
    const rect = canvasRef.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  // Find node at given world position
  const getNodeAtPosition = (worldPos: Point): GraphNode | null => {
    for (const node of props.graphData.nodes) {
      const nodeSize = (node.size || 50) / 2;
      const dx = worldPos.x - node.position.x;
      const dy = worldPos.y - node.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= nodeSize) {
        return node;
      }
    }
    return null;
  };

    // Update highlight state based on selected node and filter
  const updateHighlightState = () => {
    const highlighted = new Set<string>();
    const highlightedEdges = new Set<string>();
    const dimmed = new Set<string>();
    const dimmedEdges = new Set<string>();

    if (props.filterState.active) {
      // Highlight selected article (legacy single selection)
      if (props.filterState.articleId) {
        const articleNodeId = `article-${props.filterState.articleId}`;
        highlighted.add(articleNodeId);

        for (const edge of props.graphData.edges) {
          if (edge.source === articleNodeId) {
            highlightedEdges.add(edge.id);
            highlighted.add(edge.target);
          }
        }
      }

      // Highlight selected articles (multi-select)
      if (props.filterState.articleIds && props.filterState.articleIds.length > 0) {
        for (const articleId of props.filterState.articleIds) {
          const articleNodeId = `article-${articleId}`;
          highlighted.add(articleNodeId);

          // Find all edges from this article
          for (const edge of props.graphData.edges) {
            if (edge.source === articleNodeId) {
              highlightedEdges.add(edge.id);
              highlighted.add(edge.target);
            }
          }
        }
      }

      // Highlight selected property values and connected articles
      if (props.filterState.propertyFilters && props.filterState.propertyFilters.length > 0) {
        for (const filter of props.filterState.propertyFilters) {
          const propertyNodeId = `${filter.property}-${filter.value}`;
          highlighted.add(propertyNodeId);

          // Find all articles connected to this property value
          for (const edge of props.graphData.edges) {
            if (edge.target === propertyNodeId) {
              highlightedEdges.add(edge.id);
              highlighted.add(edge.source);
            }
          }
        }
      }

      // Legacy single property filter
      if (props.filterState.property && props.filterState.value) {
        const propertyNodeId = `${props.filterState.property}-${props.filterState.value}`;
        highlighted.add(propertyNodeId);

        for (const edge of props.graphData.edges) {
          if (edge.target === propertyNodeId) {
            highlightedEdges.add(edge.id);
            highlighted.add(edge.source);
          }
        }
      }

      // Dim all other nodes and edges if we have filters
      if (highlighted.size > 0) {
        for (const node of props.graphData.nodes) {
          if (!highlighted.has(node.id)) {
            dimmed.add(node.id);
          }
        }
        for (const edge of props.graphData.edges) {
          if (!highlightedEdges.has(edge.id)) {
            dimmedEdges.add(edge.id);
          }
        }
      }
    }

    setHighlightState({
      highlightedNodes: highlighted,
      highlightedEdges: highlightedEdges,
      dimmedNodes: dimmed,
      dimmedEdges: dimmedEdges
    });
  };

  // Watch for filter changes
  createEffect(() => {
    updateHighlightState();
  });

  // Render a single node with columnar styling
  const renderNode = (ctx: CanvasRenderingContext2D, node: GraphNode) => {
    if (!canvasRef) return;

    const screenPos = worldToScreen(node.position);
    const baseSize = node.size || 50;
    const radius = (baseSize / 2) * transform.scale;

    // Skip rendering if node is outside viewport
    const margin = radius + 50;
    if (screenPos.x < -margin || screenPos.x > canvasRef.width + margin ||
        screenPos.y < -margin || screenPos.y > canvasRef.height + margin) {
      return;
    }

    const state = highlightState();
    const isHighlighted = state.highlightedNodes.has(node.id);
    const isDimmed = state.dimmedNodes.has(node.id);

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
      const padding = 30; // Padding on each side
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
  const getArticleLineColor = (articleId: string, isHighlighted: boolean, isDimmed: boolean): string => {
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
  const renderArticleConnections = (ctx: CanvasRenderingContext2D, articleId: string) => {
    if (!canvasRef) return;

    const articleNode = props.graphData.nodes.find(n => n.id === articleId);
    if (!articleNode) return;

    // Find all edges from this article, sorted by column position
    const edges = props.graphData.edges
      .filter(e => e.source === articleId)
      .sort((a, b) => {
        const nodeA = props.graphData.nodes.find(n => n.id === a.target);
        const nodeB = props.graphData.nodes.find(n => n.id === b.target);
        return (nodeA?.position.x || 0) - (nodeB?.position.x || 0);
      });

    if (edges.length === 0) return;

    const state = highlightState();
    const isHighlighted = state.highlightedNodes.has(articleId);
    const isDimmed = state.dimmedNodes.has(articleId);

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

    const articleScreen = worldToScreen(articleNode.position);
    ctx.moveTo(articleScreen.x, articleScreen.y);

    // Draw smooth curve through each property node
    for (let i = 0; i < edges.length; i++) {
      const targetNode = props.graphData.nodes.find(n => n.id === edges[i].target);
      if (!targetNode) continue;

      const targetScreen = worldToScreen(targetNode.position);

      if (i === 0) {
        // First segment: quadratic curve from article to first property
        const controlX = (articleScreen.x + targetScreen.x) / 2;
        const controlY = articleScreen.y;
        ctx.quadraticCurveTo(controlX, controlY, targetScreen.x, targetScreen.y);
      } else {
        // Subsequent segments: smooth curve to next property
        const prevNode = props.graphData.nodes.find(n => n.id === edges[i - 1].target);
        if (!prevNode) continue;
        const prevScreen = worldToScreen(prevNode.position);
        
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
        const targetNode = props.graphData.nodes.find(n => n.id === edge.target);
        if (!targetNode) continue;
        const targetScreen = worldToScreen(targetNode.position);
        ctx.beginPath();
        ctx.arc(targetScreen.x, targetScreen.y, 5 * transform.scale, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  };

  // Main render function
  const render = () => {
    if (!canvasRef) return;
    const ctx = canvasRef.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
    
    // Set high-quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw vertical column separators
    drawColumnSeparators(ctx);

    // Render continuous lines for each article
    const articles = props.graphData.nodes.filter(n => n.metadata.columnType === 'article');
    for (const article of articles) {
      renderArticleConnections(ctx, article.id);
    }
    
    // Render nodes on top
    props.graphData.nodes.forEach(node => renderNode(ctx, node));
  };

  // Draw column separators
  const drawColumnSeparators = (ctx: CanvasRenderingContext2D) => {
    if (!canvasRef) return;

    const columns = new Set<number>();
    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType === 'header') {
        columns.add(node.position.x);
      }
    }

    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    for (const columnX of columns) {
      const screenX = worldToScreen({ x: columnX, y: 0 }).x;
      ctx.beginPath();
      ctx.moveTo(screenX, 0);
      ctx.lineTo(screenX, canvasRef.height);
      ctx.stroke();
    }

    ctx.setLineDash([]);
  };

  // Animation loop
  const animate = () => {
    render();
    animationFrameId = requestAnimationFrame(animate);
  };

  // Mouse event handlers with multi-select support (Ctrl/Cmd key)
  const handleMouseDown = (e: MouseEvent) => {
    if (!canvasRef) return;

    const mousePos = getMousePos(e);
    const worldPos = screenToWorld(mousePos);
    const node = getNodeAtPosition(worldPos);
    
    if (node) {
      selectedNode = node;
      props.onNodeSelect(node);
      
      const isMultiSelect = e.ctrlKey || e.metaKey; // Ctrl on Windows/Linux, Cmd on Mac
      
      // Update filter based on clicked node
      if (node.metadata.columnType === 'article') {
        const articleId = node.metadata.originalExpression || '';
        const currentArticleIds = new Set(props.filterState.articleIds || []);
        
        if (isMultiSelect) {
          // Toggle article in selection
          if (currentArticleIds.has(articleId)) {
            currentArticleIds.delete(articleId);
          } else {
            currentArticleIds.add(articleId);
          }
        } else {
          // Single select
          currentArticleIds.clear();
          currentArticleIds.add(articleId);
        }
        
        props.onFilterChange({
          active: currentArticleIds.size > 0 || (props.filterState.propertyFilters?.length || 0) > 0,
          articleIds: Array.from(currentArticleIds),
          propertyFilters: props.filterState.propertyFilters || []
        });
      } else if (node.metadata.columnType === 'property') {
        const propertyFilter = {
          property: node.metadata.propertyName || '',
          value: node.label
        };
        
        let currentFilters = [...(props.filterState.propertyFilters || [])];
        
        if (isMultiSelect) {
          // Toggle property filter
          const index = currentFilters.findIndex(
            f => f.property === propertyFilter.property && f.value === propertyFilter.value
          );
          if (index >= 0) {
            currentFilters.splice(index, 1);
          } else {
            currentFilters.push(propertyFilter);
          }
        } else {
          // Single select
          currentFilters = [propertyFilter];
        }
        
        props.onFilterChange({
          active: (props.filterState.articleIds?.length || 0) > 0 || currentFilters.length > 0,
          articleIds: props.filterState.articleIds || [],
          propertyFilters: currentFilters
        });
      }
    } else {
      // Clear selection and filter if not multi-selecting
      if (!(e.ctrlKey || e.metaKey)) {
        selectedNode = null;
        props.onFilterChange({
          active: false,
          articleIds: [],
          propertyFilters: []
        });
      }
      
      // Start panning
      isPanning = true;
      lastPanPoint = mousePos;
      canvasRef.style.cursor = 'grabbing';
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!canvasRef) return;

    const mousePos = getMousePos(e);
    const worldPos = screenToWorld(mousePos);
    
    if (isPanning) {
      const dx = mousePos.x - lastPanPoint.x;
      const dy = mousePos.y - lastPanPoint.y;
      
      transform.x += dx;
      transform.y += dy;
      
      lastPanPoint = mousePos;
    } else {
      const node = getNodeAtPosition(worldPos);
      hoveredNode = node;
      canvasRef.style.cursor = node ? 'pointer' : 'grab';
    }
  };

  const handleMouseUp = () => {
    if (!canvasRef) return;
    isPanning = false;
    canvasRef.style.cursor = 'grab';
  };

  const handleMouseLeave = () => {
    if (!canvasRef) return;
    isPanning = false;
    hoveredNode = null;
    canvasRef.style.cursor = 'grab';
  };

  // Zoom handling
  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();

    const mousePos = getMousePos(e);
    const worldPosBeforeZoom = screenToWorld(mousePos);

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    let newScale = transform.scale * zoomFactor;
    newScale = Math.max(0.3, Math.min(3, newScale));

    if (Math.abs(newScale - transform.scale) < 0.0001) return;

    transform.scale = newScale;

    const worldPosAfterZoom = screenToWorld(mousePos);
    transform.x += (worldPosAfterZoom.x - worldPosBeforeZoom.x) * transform.scale;
    transform.y += (worldPosAfterZoom.y - worldPosBeforeZoom.y) * transform.scale;
  };

  onMount(() => {
    if (!canvasRef) return;
    
    const resizeCanvas = () => {
      if (!canvasRef) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvasRef.getBoundingClientRect();
      
      canvasRef.width = rect.width * dpr;
      canvasRef.height = rect.height * dpr;
      
      const ctx = canvasRef.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }
      
      canvasRef.style.width = rect.width + 'px';
      canvasRef.style.height = rect.height + 'px';
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    canvasRef.addEventListener('mousedown', handleMouseDown);
    canvasRef.addEventListener('mousemove', handleMouseMove);
    canvasRef.addEventListener('mouseup', handleMouseUp);
    canvasRef.addEventListener('mouseleave', handleMouseLeave);
    canvasRef.addEventListener('wheel', handleWheel);

    animate();

    onCleanup(() => {
      if (!canvasRef) return;
      window.removeEventListener('resize', resizeCanvas);
      canvasRef.removeEventListener('mousedown', handleMouseDown);
      canvasRef.removeEventListener('mousemove', handleMouseMove);
      canvasRef.removeEventListener('mouseup', handleMouseUp);
      canvasRef.removeEventListener('mouseleave', handleMouseLeave);
      canvasRef.removeEventListener('wheel', handleWheel);
      
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    });
  });

  return (
    <canvas 
      ref={el => canvasRef = el as HTMLCanvasElement}
      class={styles.canvas}
      style={{
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        cursor: 'grab',
        'background-color': '#fafafa'
      }}
    >
      Your browser does not support the HTML5 canvas element.
    </canvas>
  );
};

export default ColumnarVisualizer;
