// Graph Visualizer component - canvas-based graph rendering
import { Component, onMount, createEffect, onCleanup } from 'solid-js';
import { GraphData, GraphNode, GraphEdge, Point } from '../../types';

export interface GraphVisualizerProps {
  graphData: GraphData;
  onNodeSelect: (node: GraphNode) => void;
  onNodeDrag: (nodeId: string, position: { x: number; y: number }) => void;
}

interface Transform {
  x: number;
  y: number;
  scale: number;
}

const GraphVisualizer: Component<GraphVisualizerProps> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let animationFrameId: number;
  
  // Canvas transformation state
  let transform: Transform = { x: 0, y: 0, scale: 1 };
  let isPanning = false;
  let isDraggingNode = false;
  let draggedNode: GraphNode | null = null;
  let dragOffset: Point = { x: 0, y: 0 };
  let lastPanPoint: Point = { x: 0, y: 0 };
  let hoveredNode: GraphNode | null = null;
  let selectedNode: GraphNode | null = null;

  // Node colors based on template.html color scheme
  const getNodeColor = (node: GraphNode): string => {
    const label = node.label || '';
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
      hash = (hash << 5) - hash + label.charCodeAt(i);
      hash |= 0; 
    }
    const hue = Math.abs(hash) % 360;
    const sat = 60 + (Math.abs(hash) % 30); // 60-89%
    const light = 45 + (Math.abs(hash) % 20); // 45-64%
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  };
  const getEdgeColor = (edge: GraphEdge): string => {
    switch (edge.type) {
      case 'relation': return 'rgba(107, 114, 128, 0.6)';
      case 'hypergraph_connection': return 'rgba(168, 85, 247, 0.6)';
      default: return 'rgba(107, 114, 128, 0.6)';
    }
  };

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

  // Find node at given world position (with z-order consideration)
  const getNodeAtPosition = (worldPos: Point): GraphNode | null => {
    const nodeRadius = 20; // Base node radius
    let foundNodes: { node: GraphNode; distance: number }[] = [];
    
    // Find all nodes under the cursor
    for (const node of props.graphData.nodes) {
      const dx = worldPos.x - node.position.x;
      const dy = worldPos.y - node.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= nodeRadius) {
        foundNodes.push({ node, distance });
      }
    }
    
    if (foundNodes.length === 0) return null;
    
    // Sort by distance (closest first) and prioritize selected node
    foundNodes.sort((a, b) => {
      // Selected node gets priority
      if (a.node === selectedNode) return -1;
      if (b.node === selectedNode) return 1;
      // Then by distance
      return a.distance - b.distance;
    });
    
    return foundNodes[0].node;
  };

  // Render a single node with interaction states
  const renderNode = (ctx: CanvasRenderingContext2D, node: GraphNode) => {
    if (!canvasRef) return;

    const screenPos = worldToScreen(node.position);
    const baseRadius = node.size || 40;
    let radius = baseRadius * transform.scale;
    
    // Skip rendering if node is outside viewport (with margin)
    const margin = radius + 50;
    if (screenPos.x < -margin || screenPos.x > canvasRef.width + margin ||
        screenPos.y < -margin || screenPos.y > canvasRef.height + margin) {
      return;
    }

    // Apply visual effects based on interaction state
    let fillColor = node.color || getNodeColor(node);
    let strokeColor = 'rgba(0, 0, 0, 0.3)';
    let strokeWidth = 1;
    
    // Hover effect
    if (node === hoveredNode) {
      radius *= 1.1; // Slightly larger
      fillColor = fillColor.replace(/0\.\d+\)$/, '0.9)'); // More opaque
      strokeColor = 'rgba(0, 0, 0, 0.5)';
      strokeWidth = 2;
    }
    
    // Selection effect
    if (node === selectedNode) {
      strokeColor = 'rgba(59, 130, 246, 0.8)'; // Blue selection ring
      strokeWidth = 3;
    }
    
    // Dragging effect
    if (node === draggedNode) {
      fillColor = fillColor.replace(/0\.\d+\)$/, '1.0)'); // Fully opaque
      strokeColor = 'rgba(34, 197, 94, 0.8)'; // Green drag ring
      strokeWidth = 3;
      
      // Add subtle shadow effect
      ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
    }

    // Draw node circle
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = fillColor;
    ctx.fill();
    
    // Draw node border
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Draw node label if scale is large enough
    if (transform.scale > 0.5 && node.label) {
      ctx.fillStyle = '#374151';
      ctx.font = `${Math.max(14, 16 * transform.scale)}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Add text background for better readability
      const textMetrics = ctx.measureText(node.label);
      const textWidth = textMetrics.width;
      const textHeight = 16 * transform.scale;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(
        screenPos.x - textWidth / 2 - 4,
        screenPos.y - textHeight / 2 - 2,
        textWidth + 8,
        textHeight + 4
      );
      
      ctx.fillStyle = '#374151';
      ctx.fillText(node.label, screenPos.x, screenPos.y);
    }
  };

  // Render a single edge
  const renderEdge = (ctx: CanvasRenderingContext2D, edge: GraphEdge) => {
    if (!canvasRef) return;

    const sourceNode = props.graphData.nodes.find(n => n.id === edge.source);
    const targetNode = props.graphData.nodes.find(n => n.id === edge.target);

    if (!sourceNode || !targetNode) return;

    const sourceScreen = worldToScreen(sourceNode.position);
    const targetScreen = worldToScreen(targetNode.position);

    // Calculate direction and node radii
    const angle = Math.atan2(targetScreen.y - sourceScreen.y, targetScreen.x - sourceScreen.x);
    const sourceRadius = (sourceNode.size || 40) * transform.scale;
    const targetRadius = (targetNode.size || 40) * transform.scale;

    // Calculate start/end points so edge stops at node borders
    const startX = sourceScreen.x + Math.cos(angle) * sourceRadius;
    const startY = sourceScreen.y + Math.sin(angle) * sourceRadius;
    const endX = targetScreen.x - Math.cos(angle) * targetRadius;
    const endY = targetScreen.y - Math.sin(angle) * targetRadius;

    // Draw edge line (stops at node borders)
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = edge.color || getEdgeColor(edge);
    ctx.lineWidth = Math.max(2, 4 * transform.scale);
    ctx.stroke();

    // Draw arrow for directed edges
    if (edge.directed) {
      const arrowLength = Math.max(8, 12 * transform.scale);
      const arrowAngle = Math.PI / 6;

      // Arrow base is at endX, endY
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle - arrowAngle),
        endY - arrowLength * Math.sin(angle - arrowAngle)
      );
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle + arrowAngle),
        endY - arrowLength * Math.sin(angle + arrowAngle)
      );
      ctx.strokeStyle = edge.color || getEdgeColor(edge);
      ctx.lineWidth = Math.max(2, 4 * transform.scale);
      ctx.stroke();
    }

    // Draw edge label if scale is large enough
    if (transform.scale > 0.7 && edge.label) {
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;

      ctx.fillStyle = '#6b7280';
      ctx.font = `${Math.max(12, 14 * transform.scale)}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Add text background
      const textMetrics = ctx.measureText(edge.label);
      const textWidth = textMetrics.width;
      const textHeight = 12 * transform.scale;

      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(
        midX - textWidth / 2 - 3,
        midY - textHeight / 2 - 1,
        textWidth + 6,
        textHeight + 2
      );

      ctx.fillStyle = '#6b7280';
      ctx.fillText(edge.label, midX, midY);
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

    // Draw background grid (optional, subtle)
    if (transform.scale > 0.3) {
      drawGrid(ctx);
    }

    // Render edges first (so they appear behind nodes)
    props.graphData.edges.forEach(edge => renderEdge(ctx, edge));
    
    // Render nodes with z-order management (selected and dragged nodes on top)
    const regularNodes = props.graphData.nodes.filter(n => n !== selectedNode && n !== draggedNode);
    const priorityNodes = props.graphData.nodes.filter(n => n === selectedNode || n === draggedNode);
    
    // Render regular nodes first
    regularNodes.forEach(node => renderNode(ctx, node));
    
    // Render priority nodes on top
    priorityNodes.forEach(node => renderNode(ctx, node));
  };

  // Draw subtle background grid
  const drawGrid = (ctx: CanvasRenderingContext2D) => {
    if (!canvasRef) return;
    const gridSize = 50 * transform.scale;
    const offsetX = transform.x % gridSize;
    const offsetY = transform.y % gridSize;
    
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.03)';
    ctx.lineWidth = 1;
    
    // Vertical lines
    for (let x = offsetX; x < canvasRef.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasRef.height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = offsetY; y < canvasRef.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasRef.width, y);
      ctx.stroke();
    }
  };

  // Animation loop
  const animate = () => {
    render();
    animationFrameId = requestAnimationFrame(animate);
  };

  // Mouse event handlers with node interaction
  const handleMouseDown = (e: MouseEvent) => {
    if (!canvasRef) return;

    const mousePos = getMousePos(e);
    const worldPos = screenToWorld(mousePos);
    const node = getNodeAtPosition(worldPos);
    
    if (node) {
      // Node interaction
      selectedNode = node;
      props.onNodeSelect(node);
      
      // Start node dragging
      isDraggingNode = true;
      draggedNode = node;
      dragOffset = {
        x: worldPos.x - node.position.x,
        y: worldPos.y - node.position.y
      };
      canvasRef.style.cursor = 'grabbing';
    } else {
      // Clear selection if clicking on empty space
      selectedNode = null;
      
      // Start canvas panning
      isPanning = true;
      lastPanPoint = mousePos;
      canvasRef.style.cursor = 'grabbing';
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!canvasRef) return;

    const mousePos = getMousePos(e);
    const worldPos = screenToWorld(mousePos);
    
    if (isDraggingNode && draggedNode) {
      // Update node position during drag
      const newPosition = {
        x: worldPos.x - dragOffset.x,
        y: worldPos.y - dragOffset.y
      };
      
      // Update the node position in the graph data
      draggedNode.position = newPosition;
      
      // Notify parent component about the drag
      props.onNodeDrag(draggedNode.id, newPosition);
      
    } else if (isPanning) {
      // Pan the canvas
      const dx = mousePos.x - lastPanPoint.x;
      const dy = mousePos.y - lastPanPoint.y;
      
      transform.x += dx;
      transform.y += dy;
      
      lastPanPoint = mousePos;
    } else {
      // Update hover state and cursor
      const node = getNodeAtPosition(worldPos);
      
      // Update hovered node
      if (hoveredNode !== node) {
        hoveredNode = node;
      }
      
      // Update cursor
      if (node) {
        canvasRef.style.cursor = 'pointer';
      } else {
        canvasRef.style.cursor = 'grab';
      }
    }
  };

  const handleMouseUp = () => {
    if (!canvasRef) return;

    // End all dragging operations
    if (isDraggingNode && draggedNode) {
      isDraggingNode = false;
      draggedNode = null;
    }
    
    if (isPanning) {
      isPanning = false;
    }
    
    // Reset cursor
    const mousePos = getMousePos(event as MouseEvent);
    const worldPos = screenToWorld(mousePos);
    const node = getNodeAtPosition(worldPos);
    canvasRef.style.cursor = node ? 'pointer' : 'grab';
  };

  const handleMouseLeave = () => {
    if (!canvasRef) return;

    // Clean up all interaction states when mouse leaves canvas
    isDraggingNode = false;
    isPanning = false;
    draggedNode = null;
    hoveredNode = null;
    canvasRef.style.cursor = 'grab';
  };

  // Zoom handling
  const handleWheel = (e: WheelEvent) => {
    e.preventDefault();

    const mousePos = getMousePos(e);
    const worldPosBeforeZoom = screenToWorld(mousePos);

    // Zoom factor
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    let newScale = transform.scale * zoomFactor;
    // Clamp scale to [0.1, 5]
    newScale = Math.max(0.1, Math.min(5, newScale));

    // Only update if scale actually changes
    if (Math.abs(newScale - transform.scale) < 0.0001) return;

    transform.scale = newScale;

    // Adjust position to zoom towards mouse
    const worldPosAfterZoom = screenToWorld(mousePos);
    transform.x += (worldPosAfterZoom.x - worldPosBeforeZoom.x) * transform.scale;    // ...existing code...
    
    // Helper to compute graph center
    function getGraphCenterAndScale(): { x: number; y: number; scale: number } {
      if (!canvasRef || props.graphData.nodes.length === 0) {
        return { x: 0, y: 0, scale: 1 };
      }
      const bounds = {
        minX: Math.min(...props.graphData.nodes.map(n => n.position.x)),
        maxX: Math.max(...props.graphData.nodes.map(n => n.position.x)),
        minY: Math.min(...props.graphData.nodes.map(n => n.position.y)),
        maxY: Math.max(...props.graphData.nodes.map(n => n.position.y))
      };
      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerY = (bounds.minY + bounds.maxY) / 2;
      // Optionally fit-to-view, but here we just center and set scale to 1
      return {
        x: canvasRef.width / 2 - centerX,
        y: canvasRef.height / 2 - centerY,
        scale: 1
      };
    }
    
    // Smooth recenter animation
    function animateRecenter() {
      if (!canvasRef) return;
      const target = getGraphCenterAndScale();
      const start = { ...transform };
      const duration = 300;
      const startTime = performance.now();
    
      function step(now: number) {
        const t = Math.min(1, (now - startTime) / duration);
        const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
        transform.x = start.x + (target.x - start.x) * ease;
        transform.y = start.y + (target.y - start.y) * ease;
        transform.scale = start.scale + (target.scale - start.scale) * ease;
        render();
        if (t < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    
    // Listen for custom 'recenter' event
    onMount(() => {
      // ...existing code...
      if (canvasRef) {
        canvasRef.addEventListener('recenter', animateRecenter);
      }
      // ...existing code...
      onCleanup(() => {
        if (canvasRef) {
          canvasRef.removeEventListener('recenter', animateRecenter);
        }
        // ...existing code...
      });
    });e;
    transform.y += (worldPosAfterZoom.y - worldPosBeforeZoom.y) * transform.scale;
  };

  // Helper to compute graph center and scale
  const getGraphCenterAndScale = (): { x: number; y: number; scale: number } => {
    if (!canvasRef || props.graphData.nodes.length === 0) {
      return { x: 0, y: 0, scale: 1 };
    }
    const bounds = {
      minX: Math.min(...props.graphData.nodes.map(n => n.position.x)),
      maxX: Math.max(...props.graphData.nodes.map(n => n.position.x)),
      minY: Math.min(...props.graphData.nodes.map(n => n.position.y)),
      maxY: Math.max(...props.graphData.nodes.map(n => n.position.y))
    };
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    
    // Calculate scale to fit graph in view with some padding
    const graphWidth = bounds.maxX - bounds.minX;
    const graphHeight = bounds.maxY - bounds.minY;
    const padding = 100;
    const scaleX = (canvasRef.width - padding * 2) / Math.max(graphWidth, 1);
    const scaleY = (canvasRef.height - padding * 2) / Math.max(graphHeight, 1);
    const fitScale = Math.min(scaleX, scaleY, 2); // Cap at 2x zoom
    
    return {
      x: canvasRef.width / 2 - centerX * fitScale,
      y: canvasRef.height / 2 - centerY * fitScale,
      scale: fitScale
    };
  };
  
  // Smooth recenter animation
  const animateRecenter = () => {
    if (!canvasRef) return;
    const target = getGraphCenterAndScale();
    const start = { ...transform };
    const duration = 300;
    const startTime = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / duration);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      transform.x = start.x + (target.x - start.x) * ease;
      transform.y = start.y + (target.y - start.y) * ease;
      transform.scale = start.scale + (target.scale - start.scale) * ease;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // Zoom functions for external control
  const zoomIn = () => {
    const mousePos = { x: canvasRef?.width || 0 / 2, y: canvasRef?.height || 0 / 2 };
    const worldPosBeforeZoom = screenToWorld(mousePos);
    
    const newScale = Math.min(5, transform.scale * 1.2);
    transform.scale = newScale;
    
    const worldPosAfterZoom = screenToWorld(mousePos);
    transform.x += (worldPosAfterZoom.x - worldPosBeforeZoom.x) * transform.scale;
    transform.y += (worldPosAfterZoom.y - worldPosBeforeZoom.y) * transform.scale;
  };

  const zoomOut = () => {
    const mousePos = { x: canvasRef?.width || 0 / 2, y: canvasRef?.height || 0 / 2 };
    const worldPosBeforeZoom = screenToWorld(mousePos);
    
    const newScale = Math.max(0.1, transform.scale * 0.8);
    transform.scale = newScale;
    
    const worldPosAfterZoom = screenToWorld(mousePos);
    transform.x += (worldPosAfterZoom.x - worldPosBeforeZoom.x) * transform.scale;
    transform.y += (worldPosAfterZoom.y - worldPosBeforeZoom.y) * transform.scale;
  };

  onMount(() => {
    if (!canvasRef) return;
    
    // Set canvas size to full viewport
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

    // Add event listeners
    canvasRef.addEventListener('mousedown', handleMouseDown);
    canvasRef.addEventListener('mousemove', handleMouseMove);
    canvasRef.addEventListener('mouseup', handleMouseUp);
    canvasRef.addEventListener('mouseleave', handleMouseLeave);
    canvasRef.addEventListener('wheel', handleWheel);

    // Add custom event listeners for external controls
    canvasRef.addEventListener('zoomIn', zoomIn);
    canvasRef.addEventListener('zoomOut', zoomOut);
    canvasRef.addEventListener('recenter', animateRecenter);

    // Center the graph initially
    if (props.graphData.nodes.length > 0) {
      const centerData = getGraphCenterAndScale();
      transform.x = centerData.x;
      transform.y = centerData.y;
      transform.scale = centerData.scale;
    }

    // Start animation loop
    animate();

    onCleanup(() => {
      if (!canvasRef) return;

      window.removeEventListener('resize', resizeCanvas);
      canvasRef.removeEventListener('mousedown', handleMouseDown);
      canvasRef.removeEventListener('mousemove', handleMouseMove);
      canvasRef.removeEventListener('mouseup', handleMouseUp);
      canvasRef.removeEventListener('mouseleave', handleMouseLeave);
      canvasRef.removeEventListener('wheel', handleWheel);
      canvasRef.removeEventListener('zoomIn', zoomIn);
      canvasRef.removeEventListener('zoomOut', zoomOut);
      canvasRef.removeEventListener('recenter', animateRecenter);
      
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    });
  });

  // React to graph data changes
  createEffect(() => {
    if (props.graphData.nodes.length > 0) {
      // Recenter when new data is loaded
      const bounds = {
        minX: Math.min(...props.graphData.nodes.map(n => n.position.x)),
        maxX: Math.max(...props.graphData.nodes.map(n => n.position.x)),
        minY: Math.min(...props.graphData.nodes.map(n => n.position.y)),
        maxY: Math.max(...props.graphData.nodes.map(n => n.position.y))
      };
      
      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerY = (bounds.minY + bounds.maxY) / 2;
      
      transform.x = (canvasRef?.width || window.innerWidth) / 2 - centerX * transform.scale;
      transform.y = (canvasRef?.height || window.innerHeight) / 2 - centerY * transform.scale;
    }
  });

  return (
    <canvas 
      ref={el => canvasRef = el as HTMLCanvasElement}
      id="graph-canvas"
      style={{
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100vw',
        height: '100vh',
        cursor: 'grab',
        'background-color': 'var(--bg-primary)',
        'background-image': 'radial-gradient(circle at 1px 1px, rgba(0,0,0,0.02) 1px, transparent 0)',
        'background-size': '20px 20px'
      }}
    >
      Your browser does not support the HTML5 canvas element.
    </canvas>
  );
};

export default GraphVisualizer;