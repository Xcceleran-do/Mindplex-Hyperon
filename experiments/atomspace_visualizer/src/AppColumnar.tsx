import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource } from 'solid-js';

import ColumnarVisualizer from './components/ColumnarVisualizer/ColumnarVisualizer';
import EnhancedLegend from './components/Legend/EnhancedLegend';
import EnhancedUIControls from './components/UIControls/EnhancedUIControls';
import MiningPanel from './components/MiningPanel/MiningPanel';
import { GraphData, GraphNode, FilterState } from './types';
import { MettaParserImpl } from './services/parser/MettaParser';
import { ColumnarTransformer } from './services/graph/ColumnarTransformer';

import './styles/variables.css';
import './styles/components.css';
import styles from './App.module.css';

const App: Component = () => {
  // Load initial text from small-ugly.metta file
  const [initialTextResource] = createResource(async () => {
    try {
      const response = await fetch('/small-ugly.metta');
      if (!response.ok) {
        throw new Error('Failed to load file');
      }
      const text = await response.text();
      return text;
    } catch (error) {
      console.error('Error loading initial text:', error);
      return '';
    }
  });

  // Core application state
  const [mettaText, setMettaText] = createSignal('');
  
  // Set initial text when resource loads
  createEffect(() => {
    const loadedText = initialTextResource();
    if (loadedText) {
      setMettaText(loadedText);
    }
  });

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

  const [filterState, setFilterState] = createSignal<FilterState>({
    active: false,
    articleIds: [],
    propertyFilters: []
  });

  // Track collapsed states for dynamic positioning
  const [controlsCollapsed, setControlsCollapsed] = createSignal(true);
  const [legendCollapsed, setLegendCollapsed] = createSignal(true);

  // Initialize parser and columnar transformer
  const parser = new MettaParserImpl();
  const columnarTransformer = new ColumnarTransformer();

  // Update CSS variables for dynamic positioning
  createEffect(() => {
    const updatePositions = () => {
      const controlsEl = document.querySelector('[class*="controlsContainer"]') as HTMLElement;
      const legendEl = document.querySelector('[class*="legendContainer"]') as HTMLElement;
      
      if (controlsEl && legendEl) {
        const controlsHeight = controlsEl.offsetHeight;
        const legendTop = 20 + controlsHeight;
        const legendHeight = legendEl.offsetHeight;
        const miningTop = legendTop + legendHeight;
        
        document.documentElement.style.setProperty('--legend-top', `${legendTop}px`);
        document.documentElement.style.setProperty('--mining-top', `${miningTop}px`);
      }
    };

    // Update on mount and when layout changes
    setTimeout(updatePositions, 100);
    const observer = new MutationObserver(updatePositions);
    observer.observe(document.body, { 
      childList: true, 
      subtree: true, 
      attributes: true, 
      attributeFilter: ['class', 'style'] 
    });

    return () => observer.disconnect();
  });

  // Parse and transform data to columnar format
  createEffect(() => {
    if (mettaText().trim()) {
      try {
        const parseResult = parser.parse(mettaText());
        const triples = parser.extractTriples(mettaText());
        const columnarData = columnarTransformer.transformToColumnar(triples);
        setGraphData(columnarData);
      } catch (error) {
        console.error('Parsing error:', error);
      }
    }
  });

  // Event handlers
  const handleNodeSelect = (node: GraphNode) => {
    console.log('Selected node:', node.label);
  };

  const handleFilterChange = (filter: FilterState) => {
    setFilterState(filter);
  };

  const handleZoomIn = () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const evt = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: -120,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
    canvas.dispatchEvent(evt);
  };

  const handleZoomOut = () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const evt = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      deltaY: 120,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
    canvas.dispatchEvent(evt);
  };

  const handleReset = () => {
    setFilterState({
      active: false,
      articleIds: [],
      propertyFilters: []
    });
    // Reset view by reloading
    const canvas = document.querySelector('canvas');
    if (canvas) {
      canvas.dispatchEvent(new CustomEvent('reset'));
    }
  };

  return (
    <div class={styles.app}>
      {/* Scrollable graph container */}
      <div class={styles.graphContainer}>
        <div class={styles.graphCard}>
          <ColumnarVisualizer
            graphData={graphData()}
            onNodeSelect={handleNodeSelect}
            filterState={filterState()}
            onFilterChange={handleFilterChange}
          />
        </div>
      </div>

      {/* Enhanced UI Controls - Top Right */}
      <EnhancedUIControls
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onReset={handleReset}
      />

      {/* Enhanced Legend - Below Controls */}
      <EnhancedLegend
        graphData={graphData()}
        onFilterChange={handleFilterChange}
        filterState={filterState()}
      />

      {/* Mining Panel - Below Legend */}
      <MiningPanel
        onFilterChange={handleFilterChange}
      />
    </div>
  );
};

export default App;
