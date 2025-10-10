// Columnar Visualizer component - property-based columnar layout
import { Component, onMount, createEffect, onCleanup, createSignal } from 'solid-js';
import { GraphData, GraphNode, Point, FilterState, HighlightState } from '../../types';
import styles from './ColumnarVisualizer.module.css';
import {
  Transform,
  screenToWorld,
  worldToScreen,
  renderNode,
  renderArticleConnections,
  drawColumnSeparators
} from '../../utils/canvasRenderer';
import {
  getMousePos,
  getNodeAtPosition,
  handleZoom as handleZoomUtil,
  handlePan as handlePanUtil
} from '../../utils/canvasInteractions';
import { updateHighlightState as updateHighlightStateUtil } from '../../utils/highlightManager';

export interface ColumnarVisualizerProps {
  graphData: GraphData;
  onNodeSelect: (node: GraphNode) => void;
  filterState: FilterState;
  onFilterChange: (filter: FilterState) => void;
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

  // Watch for filter changes
  createEffect(() => {
    const newState = updateHighlightStateUtil(props.graphData, props.filterState);
    setHighlightState(newState);
  });



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
    drawColumnSeparators(ctx, props.graphData, transform, canvasRef.height);

    // Render continuous lines for each article
    const articles = props.graphData.nodes.filter(n => n.metadata.columnType === 'article');
    for (const article of articles) {
      renderArticleConnections(ctx, article.id, props.graphData, transform, highlightState());
    }
    
    // Render nodes on top
    props.graphData.nodes.forEach(node => 
      renderNode(ctx, node, transform, highlightState(), hoveredNode, selectedNode, canvasRef!.width, canvasRef!.height)
    );
  };

  // Animation loop
  const animate = () => {
    render();
    animationFrameId = requestAnimationFrame(animate);
  };

  // Mouse event handlers with multi-select support (Ctrl/Cmd key)
  const handleMouseDown = (e: MouseEvent) => {
    if (!canvasRef) return;

    const mousePos = getMousePos(e, canvasRef);
    const worldPos = screenToWorld(mousePos, transform);
    const node = getNodeAtPosition(worldPos, props.graphData.nodes);
    
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

    const mousePos = getMousePos(e, canvasRef);
    const worldPos = screenToWorld(mousePos, transform);
    
    if (isPanning) {
      const dx = mousePos.x - lastPanPoint.x;
      const dy = mousePos.y - lastPanPoint.y;
      
      transform = handlePanUtil(dx, dy, transform);
      lastPanPoint = mousePos;
    } else {
      const node = getNodeAtPosition(worldPos, props.graphData.nodes);
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
    if (!canvasRef) return;
    transform = handleZoomUtil(e, canvasRef, transform);
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
