import type { Component } from 'solid-js';
import { createSignal, createEffect, Show, onMount } from 'solid-js';

// Remove MettaEditor import
import GraphVisualizer from './components/GraphVisualizer/GraphVisualizer';
import Legend from './components/Legend/Legend';
import ContextMenu from './components/ContextMenu/ContextMenu';
import UIControls from './components/UIControls/UIControls';
import MiningInterface from './components/MiningInterface/MiningInterface';
import ChatInterface from './components/ChatInterface/ChatInterface';
import { GraphData, GraphNode, ParseError, LayoutAlgorithm, LayoutOptions, LayoutState } from './types';
import { MettaParserImpl } from './services/parser/MettaParser';
import { GraphEngineImpl } from './services/graph/GraphEngine';

import './styles/variables.css';
import './styles/components.css';

const App: Component = () => {
  // Core application state - mettaText is populated by mining results only
  const [mettaText, setMettaText] = createSignal('');
  const [graphData, setGraphData] = createSignal<GraphData>({
    nodes: [],
    edges: [],
    metadata: {
      nodeCount: 0,
      edgeCount: 0,
      hypergraphCount: 0,
      lastUpdated: new Date()
    },
    hypergraphs: []
  });
  const [parseErrors, setParseErrors] = createSignal<ParseError[]>([]);

  // UI state management
  const [contextMenuNode, setContextMenuNode] = createSignal<GraphNode | null>(null);
  const [contextMenuPosition, setContextMenuPosition] = createSignal<{ x: number; y: number } | null>(null);
  const [showLabels, setShowLabels] = createSignal(true);
  const [isLoading, setIsLoading] = createSignal(false);

  // Minimize state management
  const [isLegendMinimized, setIsLegendMinimized] = createSignal(false);
  const [isControlsMinimized, setIsControlsMinimized] = createSignal(false);

  // Layout state management
  const [layoutState, setLayoutState] = createSignal<LayoutState>({
    isAnimating: false,
    progress: 0,
    algorithm: 'hierarchical',
    startTime: 0,
    duration: 0
  });

  // Chat and mining results state
  const [miningResults, setMiningResults] = createSignal<Array<{ pattern: string; support: string }>>([]);
  const [filteredPattern, setFilteredPattern] = createSignal<string | null>(null);
  const [currentConjunctSize, setCurrentConjunctSize] = createSignal<number | undefined>(undefined);

  // Initialize parser and graph engine
  const parser = new MettaParserImpl();
  const graphEngine = new GraphEngineImpl();

  // Signal to trigger legend position updates
  const [legendPositionTrigger, setLegendPositionTrigger] = createSignal(0);

  // Monitor controls card size changes to update legend position
  createEffect(() => {
    const controlsCard = document.getElementById('controls-card');
    if (!controlsCard) return;

    const resizeObserver = new ResizeObserver(() => {
      setLegendPositionTrigger(prev => prev + 1);
    });

    resizeObserver.observe(controlsCard);

    return () => {
      resizeObserver.disconnect();
    };
  });

  // Parse initial text on component mount
  const EDGE_LENGTH_SCALE = 2;
  createEffect(() => {
    if (mettaText().trim()) {
      setIsLoading(true);
      try {
        const parseResult = parser.parse(mettaText());
        const scaledNodes = parseResult.nodes.map(node => ({
          ...node,
          position: {
            x: node.position.x * EDGE_LENGTH_SCALE,
            y: node.position.y * EDGE_LENGTH_SCALE,
          }
        }));
        setGraphData({
          nodes: scaledNodes,
          edges: parseResult.edges,
          metadata: parseResult.metadata,
          hypergraphs: []
        });
        setParseErrors(parseResult.errors);

        // Auto-apply hierar
        if (scaledNodes.length > 0) {
          handleApplyLayout('hierarchical');
        }
      } catch (error) {
        console.error('Parsing error:', error);
        setParseErrors([{
          line: 1,
          column: 1,
          message: 'Failed to parse Metta text',
          severity: 'error'
        }]);
      } finally {
        setIsLoading(false);
      }
    } else {
      // Clear graph when text is empty
      setGraphData({
        nodes: [],
        edges: [],
        metadata: {
          nodeCount: 0,
          edgeCount: 0,
          hypergraphCount: 0,
          lastUpdated: new Date()
        },
        hypergraphs: []
      });
      setParseErrors([]);
    }
  });

  // Event handlers for UI interactions
  const handleTextChange = (text: string) => {
    setMettaText(text);
    // Parsing is handled by the createEffect above
  };

  const handleFileUpload = (file: File) => {
    // TODO: Implement file upload
    console.log('File uploaded:', file.name);
  };

  const handleNodeSelect = (node: GraphNode) => {
    return;
  };

  const handleNodeDrag = (nodeId: string, position: { x: number; y: number }) => {
    return;
  };

  const handleZoomIn = () => {
    // Dispatch a synthetic wheel event to the canvas center (negative deltaY zooms in)
    const canvas = document.getElementById('graph-canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const evt = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: -120,
      clientX: cx,
      clientY: cy,
    });
    canvas.dispatchEvent(evt);
  };

  const handleZoomOut = () => {
    const canvas = document.getElementById('graph-canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const evt = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: 120,
      clientX: cx,
      clientY: cy,
    });
    canvas.dispatchEvent(evt);
  };

  const handleRecenter = () => {
    // Dispatch a custom event or call a method on GraphVisualizer to recenter
    const canvas = document.getElementById('graph-canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    canvas.dispatchEvent(new CustomEvent('recenter'));
  };

  const handleToggleLabels = (show: boolean) => {
    setShowLabels(show);
  };



  const handleApplyLayout = (algorithm: LayoutAlgorithm, options?: LayoutOptions) => {
    const currentGraphData = graphData();
    if (currentGraphData.nodes.length === 0) return;

    // Update graph engine with current data
    graphEngine.setData(currentGraphData.nodes, currentGraphData.edges);

    // Apply layout
    graphEngine.applyLayout(algorithm, options);

    // Start monitoring layout state
    const monitorLayout = () => {
      const state = graphEngine.getLayoutState();
      setLayoutState(state);

      if (state.isAnimating) {
        requestAnimationFrame(monitorLayout);
      }
    };

    monitorLayout();
  };

  const handleStopLayout = () => {
    graphEngine.stopLayout();
    setLayoutState(graphEngine.getLayoutState());
  };

  const handleExportPDF = () => {
    // TODO
    console.log('Export PDF');
  };

  const handleExportPNG = () => {
    // TODO
    console.log('Export PNG');
  };

  // Minimize handlers
  const handleToggleLegend = () => {
    setIsLegendMinimized(!isLegendMinimized());
  };

  const handleToggleControls = () => {
    setIsControlsMinimized(!isControlsMinimized());
    // Trigger legend position update after a short delay to allow DOM to update
    setTimeout(() => setLegendPositionTrigger(prev => prev + 1), 100);
  };

  const handleMinimizeAll = () => {
    setIsLegendMinimized(true);
    setIsControlsMinimized(true);
  };

  const handleMaximizeAll = () => {
    setIsLegendMinimized(false);
    setIsControlsMinimized(false);
  };

  // Calculate dynamic positions for legend and controls
  const getControlsCardTop = () => {
    // Get the actual top position of the controls card by measuring from bottom
    // Controls card is always at bottom: 20px, so its top is at bottom + height
    const controlsElement = document.getElementById('controls-card');
    if (controlsElement) {
      const rect = controlsElement.getBoundingClientRect();
      return window.innerHeight - rect.top;
    }
    // Fallback calculation when element not available
    return isControlsMinimized() ? 70 : 240; // 20px bottom + estimated height + padding
  };

  const getLegendBottomPosition = () => {
    const controlsTop = getControlsCardTop();
    const margin = 5; // 5px margin between legend bottom and controls top
    return controlsTop + margin;
  };

  const getLegendMaxHeight = () => {
    if (isLegendMinimized()) return '50px';
    const bottomPosition = getLegendBottomPosition();
    const topMargin = 100; // Leave space for title and other UI elements
    return `calc(100vh - ${bottomPosition + topMargin}px)`;
  };

  const handleIsolateNode = (node: GraphNode) => {
    console.log('Isolating node:', node.label);
    // TODO
  };

  const handleCopyLabel = (node: GraphNode) => {
    navigator.clipboard.writeText(node.label).then(() => {
      console.log('Copied label:', node.label);
    }).catch(err => {
      console.error('Failed to copy label:', err);
    });
  };

  const handleCloseContextMenu = () => {
    setContextMenuNode(null);
    setContextMenuPosition(null);
  };

  const handlePatternsFound = (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => {
    setMiningResults(patterns);
    if (conjunctSize) {
      setCurrentConjunctSize(conjunctSize);
    }
  };

  const handleVisualize = (patternStr: string) => {
    setFilteredPattern(patternStr);
    
    // Parse the pattern to extract property-value pairs
    // Pattern format: ((property1 $x "value1") (property2 $x "value2") ...)
    const properties = parsePattern(patternStr);
    
    console.log('Visualizing pattern:', patternStr);
    console.log('Extracted properties:', properties);
    
    // Get the current graph data (assume we're re-parsing from mettaText to get fresh data)
    const text = mettaText();
    if (!text) return;
    
    // Re-parse to get complete data
    const parser = new MettaParserImpl();
    const result = parser.parse(text);
    const fullData = result.data;
    
    // Build a map: article_id -> { property -> value }
    const articleProperties = new Map<string, Map<string, string>>();
    
    // Parse the graph data to extract article properties
    // Format in graph: edges from property node to article_id node, then to value node
    // But in MeTTa: (property article_id "value")
    
    // Let's extract from edges directly
    fullData.edges.forEach(edge => {
      // Edge structure: source -> target with label
      // We're looking for patterns like: property_node -> article_id_node -> value_node
      const sourceNode = fullData.nodes.find(n => n.id === edge.source);
      const targetNode = fullData.nodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode) {
        // Check if source is a property name (like "length", "tone", etc)
        // and target is a number (article ID)
        const propertyName = sourceNode.label;
        const articleId = targetNode.label;
        
        // Now find the value for this property-article combination
        // Look for edges from the article ID node
        fullData.edges.forEach(valueEdge => {
          if (valueEdge.source === edge.target) {
            const valueNode = fullData.nodes.find(n => n.id === valueEdge.target);
            if (valueNode) {
              if (!articleProperties.has(articleId)) {
                articleProperties.set(articleId, new Map());
              }
              articleProperties.get(articleId)!.set(propertyName, valueNode.label);
            }
          }
        });
      }
    });
    
    console.log('Article properties map:', articleProperties);
    
    // Find article IDs that match ALL conditions
    const matchingArticleIds = new Set<string>();
    
    articleProperties.forEach((props, articleId) => {
      let matchesAll = true;
      
      for (const [propName, propValue] of Object.entries(properties)) {
        const articleValue = props.get(propName);
        if (articleValue !== propValue) {
          matchesAll = false;
          break;
        }
      }
      
      if (matchesAll && Object.keys(properties).length > 0) {
        matchingArticleIds.add(articleId);
      }
    });
    
    console.log('Matching article IDs:', matchingArticleIds);
    
    // Now filter the graph to show only matching articles and their properties
    const matchingNodeIds = new Set<string>();
    
    if (matchingArticleIds.size > 0) {
      // Add all nodes that are part of the matching articles
      fullData.nodes.forEach(node => {
        if (matchingArticleIds.has(node.label)) {
          matchingNodeIds.add(node.id);
        }
      });
      
      // Add property and value nodes connected to matching articles
      fullData.edges.forEach(edge => {
        const sourceNode = fullData.nodes.find(n => n.id === edge.source);
        const targetNode = fullData.nodes.find(n => n.id === edge.target);
        
        if (targetNode && matchingArticleIds.has(targetNode.label)) {
          // Add property node and article node
          matchingNodeIds.add(edge.source);
          matchingNodeIds.add(edge.target);
          
          // Add value nodes
          fullData.edges.forEach(valueEdge => {
            if (valueEdge.source === edge.target) {
              matchingNodeIds.add(valueEdge.target);
            }
          });
        }
      });
      
      // Create filtered graph data
      const filteredNodes = fullData.nodes.filter(n => matchingNodeIds.has(n.id));
      const filteredEdges = fullData.edges.filter(e => 
        matchingNodeIds.has(e.source) && matchingNodeIds.has(e.target)
      );
      
      console.log(`Filtered to ${filteredNodes.length} nodes and ${filteredEdges.length} edges`);
      
      setGraphData({
        ...fullData,
        nodes: filteredNodes,
        edges: filteredEdges,
        metadata: {
          ...fullData.metadata,
          nodeCount: filteredNodes.length,
          edgeCount: filteredEdges.length
        }
      });
      
      // Re-apply layout to the filtered graph
      setTimeout(() => handleApplyLayout('force'), 100);
    } else {
      alert('No articles match all the specified conditions.');
    }
  };

  const parsePattern = (pattern: string): Record<string, string> => {
    // Parse pattern like: ((tone $x "Analytical") (length $x "low") (engagement_level $x "high"))
    const properties: Record<string, string> = {};
    const regex = /\((\w+)\s+\$\w+\s+"([^"]+)"\)/g;
    let match;
    
    while ((match = regex.exec(pattern)) !== null) {
      properties[match[1]] = match[2];
    }
    
    return properties;
  };

  return (
    <div class="app">
      {/* Full-screen canvas container - matches template.html structure */}
      <div id="canvas-container">
        <GraphVisualizer
          graphData={graphData()}
          onNodeSelect={handleNodeSelect}
          onNodeDrag={handleNodeDrag}
        />
      </div>

      {/* Floating UI Components - positioned exactly as in template.html */}

      {/* Title bar at center-top */}
      <div id="title-bar" class="ui-card top-center">
        <div id="graph-title">AtomSpace Visualizer</div>
        <div class="subtitle" id="graph-stats">
          {isLoading() ? 'Parsing...' : `${graphData().metadata.nodeCount} nodes, ${graphData().metadata.edgeCount} edges`}
        </div>
      </div>

      {/* Zoom controls in top-right corner */}
      <div id="zoom-controls" class="ui-card top-right">
        <button id="zoom-in" title="Zoom In" onClick={handleZoomIn}>+</button>
        <button id="zoom-out" title="Zoom Out" onClick={handleZoomOut}>−</button>
        <button id="recenter" title="Recenter" onClick={handleRecenter}>⌂</button>
      </div>

      {/* Mining Interface */}
      <div class="ui-card bottom-center-mining">
        <MiningInterface onPatternsFound={handlePatternsFound} />
      </div>

      {/* Chat Interface */}
      <ChatInterface 
        onVisualize={handleVisualize}
        miningResults={miningResults()}
        conjunctSize={currentConjunctSize()}
      />

      {/* Minimize/Maximize controls in top-right corner */}
      <div id="minimize-controls" class="ui-card top-right-secondary">
        <button title="Minimize All" onClick={handleMinimizeAll}>⊟</button>
        <button title="Maximize All" onClick={handleMaximizeAll}>⊞</button>
      </div>

      {/* Legend card in bottom-right area (upper) */}
      <div 
        id="legend-card" 
        class={`ui-card ${isLegendMinimized() ? 'minimized' : ''}`}
        style={`
          position: absolute;
          right: 20px;
          width: 280px;
          bottom: ${getLegendBottomPosition()}px;
          max-height: ${getLegendMaxHeight()};
        `}
        data-trigger={legendPositionTrigger()}
      >
        <div class="card-header">
          <h3>Legend</h3>
          <button class="minimize-btn" onClick={handleToggleLegend}>
            {isLegendMinimized() ? '□' : '−'}
          </button>
        </div>
        <Show when={!isLegendMinimized()}>
          <div class="card-content">
            <Legend graphData={graphData()} />
          </div>
        </Show>
      </div>

      {/* UI Controls card in bottom-right area (lower) */}
      <div 
        id="controls-card" 
        class={`ui-card ${isControlsMinimized() ? 'minimized' : ''}`}
        style={`
          position: absolute;
          right: 20px;
          width: 280px;
          bottom: 20px;
        `}
      >
        <div class="card-header">
          <h3>Controls</h3>
          <button class="minimize-btn" onClick={handleToggleControls}>
            {isControlsMinimized() ? '□' : '−'}
          </button>
        </div>
        <Show when={!isControlsMinimized()}>
          <div class="card-content">
            <UIControls
              onExportPDF={handleExportPDF}
              onExportPNG={handleExportPNG}
              showLabels={showLabels()}
              onToggleLabels={handleToggleLabels}
              onApplyLayout={handleApplyLayout}
              layoutState={layoutState()}
              onStopLayout={handleStopLayout}
            />
          </div>
        </Show>
      </div>

      {/* Context Menu - positioned dynamically */}
      <ContextMenu
        node={contextMenuNode()}
        position={contextMenuPosition()}
        onIsolate={handleIsolateNode}
        onCopyLabel={handleCopyLabel}
        onClose={handleCloseContextMenu}
      />
    </div>
  );
};

export default App;