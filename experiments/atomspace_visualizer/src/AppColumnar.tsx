import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource } from 'solid-js';

import ColumnarVisualizer from './components/ColumnarVisualizer/ColumnarVisualizer';
import EnhancedLegend from './components/Legend/EnhancedLegend';
import MiningInterface from './components/MiningInterface/MiningInterface';
import ChatInterface from './components/ChatInterface/ChatInterface';
import { GraphData, GraphNode, FilterState } from './types';
import { MettaParserImpl } from './services/parser/MettaParser';
import { ColumnarTransformer } from './services/graph/ColumnarTransformer';

import './styles/variables.css';
import './styles/components.css';
import styles from './AppColumnar.module.css';

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

  // Mining and chat state
  const [miningResults, setMiningResults] = createSignal<Array<{ pattern: string; support: string }>>([]);
  const [currentConjunctSize, setCurrentConjunctSize] = createSignal<number | undefined>(undefined);
  
  // Animation state
  let animationInterval: number | undefined;

  // Initialize parser and columnar transformer
  const parser = new MettaParserImpl();
  const columnarTransformer = new ColumnarTransformer();

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

  const handleMiningStart = () => {
    console.log('AppColumnar.tsx: Mining started, starting animation');
    startMiningAnimation();
  };

  // Unified mining flow: called by either the mining button or the chat trigger
  const startMiningUnified = async (conjunctSize: number) => {
    console.log('AppColumnar.tsx: startMiningUnified called with', conjunctSize);
    const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

    try {
      // Start the animation and visual indicator
      startMiningAnimation();

      const resp = await fetch(`${API_BASE}/api/mine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conjunction_count: conjunctSize })
      });

      if (!resp.ok) {
        throw new Error(`Mining API error ${resp.status}`);
      }

      const job = await resp.json();

      // Stop animation and set results so ChatInterface will analyze and render summaries
      stopMiningAnimation();
      setMiningResults(job.result || []);
      setCurrentConjunctSize(conjunctSize);

    } catch (err) {
      console.error('startMiningUnified error', err);
      stopMiningAnimation();
    }
  };

  const handlePatternsFound = (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => {
    console.log('AppColumnar.tsx handlePatternsFound called with:', { patterns, conjunctSize });
    
    // Stop animation when mining completes
    stopMiningAnimation();
    
    setMiningResults(patterns);
    if (conjunctSize) {
      console.log('AppColumnar.tsx setting currentConjunctSize to:', conjunctSize);
      setCurrentConjunctSize(conjunctSize);
    }
  };

  const startMiningAnimation = () => {
    // Stop any existing animation
    if (animationInterval) {
      clearInterval(animationInterval);
    }
    
    // Get all article nodes from graph data
    const articles: string[] = [];
    for (const node of graphData().nodes) {
      if (node.metadata.columnType === 'article') {
        articles.push(node.metadata.originalExpression || node.label);
      }
    }
    
    if (articles.length === 0) return;
    
    // Cycle through articles with time gap
    let currentIndex = 0;
    const intervalTime = 1000; // 1000ms between highlights
    
    animationInterval = setInterval(() => {
      // Loop back to start when reaching the end
      currentIndex = currentIndex % articles.length;
      
      // Highlight current article
      const currentArticle = articles[currentIndex];
      handleFilterChange({
        active: true,
        articleIds: [currentArticle],
        propertyFilters: []
      });
      
      currentIndex++;
    }, intervalTime) as unknown as number;
  };

  const stopMiningAnimation = () => {
    console.log('AppColumnar.tsx: Stopping mining animation');
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = undefined;
    }
    
    // Reset filter state
    handleFilterChange({
      active: false,
      articleIds: [],
      propertyFilters: []
    });
  };

  const handleVisualize = (filterState: FilterState | string) => {
    if (typeof filterState === 'string') {
      // Parse the pattern to extract property filters
      const propertyFilters: Array<{ property: string; value: string }> = [];
      const regex = /\((\w+)\s+\$\w+\s+"([^"]+)"\)/g;
      let match;
      while ((match = regex.exec(filterState)) !== null) {
        propertyFilters.push({
          property: match[1],
          value: match[2]
        });
      }
      if (propertyFilters.length > 0) {
        handleFilterChange({
          active: true,
          articleIds: [],
          propertyFilters
        });
      }
    } else {
      // Already a FilterState object
      handleFilterChange(filterState);
    }
  };

  return (
    <div class={styles.app}>
      {/* Scrollable graph container */}
      <div class={styles.graphContainer}>
        <div class={styles.graphCard}>
          {/* Canvas Control Buttons - Top of Canvas */}
          <div class={styles.canvasControls}>
            <button class={styles.controlBtn} onClick={handleZoomIn} title="Zoom In">
              🔍+
            </button>
            <button class={styles.controlBtn} onClick={handleZoomOut} title="Zoom Out">
              🔍-
            </button>
            <button class={styles.controlBtn} onClick={handleReset} title="Reset View">
              🔄
            </button>
          </div>

          <ColumnarVisualizer
            graphData={graphData()}
            onNodeSelect={handleNodeSelect}
            filterState={filterState()}
            onFilterChange={handleFilterChange}
          />
        </div>
      </div>

      {/* Enhanced Legend - Top Right */}
      <EnhancedLegend
        graphData={graphData()}
        onFilterChange={handleFilterChange}
        filterState={filterState()}
      />

      {/* Mining Interface - Below Legend (with chat integration) */}
      <MiningInterface
        onPatternsFound={handlePatternsFound}
        onMiningStart={startMiningUnified}
      />

      {/* Chat Interface - Opens automatically when mining completes */}
      <ChatInterface
        conjunctSize={currentConjunctSize()}
        onVisualize={handleVisualize}
        miningResults={miningResults()}
        onMiningStart={startMiningUnified}
      />
    </div>
  );
};

export default App;
