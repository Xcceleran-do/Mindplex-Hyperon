import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource } from 'solid-js';

import ColumnarVisualizer from './components/ColumnarVisualizer/ColumnarVisualizer';
import EnhancedLegend from './components/Legend/EnhancedLegend';
import ChatInterface from './components/ChatInterface/ChatInterface';
import MiningInterface from './components/MiningInterface/MiningInterface';
import { GraphData, GraphNode, FilterState } from './types';
import { MettaParserImpl } from './services/parser/MettaParser';
import { ColumnarTransformer } from './services/graph/ColumnarTransformer';

import './styles/variables.css';
import './styles/components.css';
import './styles/themes.css';
import styles from './AppColumnar.module.css';

const App: Component = () => {
  // Load initial text from data.metta file
  const [initialTextResource] = createResource(async () => {
    try {
      const response = await fetch('/data.metta');
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

  // Chat visibility & theme
  const [isChatOpen, setIsChatOpen] = createSignal(false);
  const [theme, setTheme] = createSignal<string>(localStorage.getItem('theme') || 'auto');

  const applyTheme = (t: string) => {
    if (t === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else if (t === 'light') {
      document.documentElement.removeAttribute('data-theme');
      document.documentElement.setAttribute('data-theme', 'light');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    localStorage.setItem('theme', t);
    setTheme(t);
  };

  // Initialize theme on mount
  createEffect(() => {
    applyTheme(theme());
  });

  // Open chat automatically when mining results arrive
  createEffect(() => {
    const results = miningResults();
    if (results && results.length > 0) {
      setIsChatOpen(true);
    }
  });

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

  const handleMiningStart = () => {
    console.log('AppColumnar.tsx: Mining started, starting animation');
    startMiningAnimation();
  };

  // Unified mining flow: called by either the mining button or the chat trigger
  const startMiningUnified = async (conjunctSize: number) => {
    console.log('AppColumnar.tsx: startMiningUnified called with', conjunctSize);
    const API_BASE = import.meta.env.VITE_API_BASE_URL || 'https://super-duper-winner-97q9rxp6vvx9hxq5q-5000.app.github.dev';

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

  // Default to collapsed (closed) as requested
  const [isRightPanelOpen, setIsRightPanelOpen] = createSignal(false);

  // Draggable left panel position (persisted)
  const getInitialPanelPos = (): { x: number; y: number } => {
    try {
      const raw = localStorage.getItem('leftPanelPos');
      if (raw) return JSON.parse(raw);
    } catch (e) {
      // ignore parse errors
    }
    return { x: 20, y: 20 };
  };
  const [panelPos, setPanelPos] = createSignal<{ x: number; y: number }>(getInitialPanelPos());

  // Drag state kept in closure
  const dragState: {
    dragging: boolean;
    offsetX: number;
    offsetY: number;
  } = { dragging: false, offsetX: 0, offsetY: 0 };

  const startPanelDrag = (e: PointerEvent) => {
    // Only start drag for primary button
    if ((e as any).button && (e as any).button !== 0) return;
    e.preventDefault();
    dragState.dragging = true;
    const el = (e.currentTarget as HTMLElement) || null;
    const rect = el?.getBoundingClientRect();
    if (rect) {
      dragState.offsetX = e.clientX - rect.left;
      dragState.offsetY = e.clientY - rect.top;
    } else {
      dragState.offsetX = 0;
      dragState.offsetY = 0;
    }
    window.addEventListener('pointermove', onPanelPointerMove);
    window.addEventListener('pointerup', endPanelDrag);
  };

  const onPanelPointerMove = (e: PointerEvent) => {
    if (!dragState.dragging) return;
    let nx = e.clientX - dragState.offsetX;
    let ny = e.clientY - dragState.offsetY;
    // keep panel within viewport bounds (small margin)
    const margin = 8;
    nx = Math.max(margin, Math.min(nx, window.innerWidth - 120));
    ny = Math.max(margin, Math.min(ny, window.innerHeight - 80));
    setPanelPos({ x: nx, y: ny });
  };

  const endPanelDrag = () => {
    if (!dragState.dragging) return;
    dragState.dragging = false;
    window.removeEventListener('pointermove', onPanelPointerMove);
    window.removeEventListener('pointerup', endPanelDrag);
    try {
      localStorage.setItem('leftPanelPos', JSON.stringify(panelPos()));
    } catch (e) {
      // ignore
    }
  };

  return (
    <div class={`${styles.app} ${isRightPanelOpen() ? styles.panelOpen : ''}`}>
      {/* Scrollable graph container */}
      <div class={styles.graphContainer}>
        <div class={styles.graphCard}>
          {/* Canvas Control Buttons - Top of Canvas */}
          <div class={styles.canvasControls}>
            <button
              class={styles.controlBtn}
              onClick={handleZoomIn}
              title="Zoom in"
              aria-label="Zoom in"
            >
              <span aria-hidden="true" class={styles.controlBtnIcon}>
                <svg
                  viewBox="0 0 24 24"
                  role="presentation"
                >
                  <circle cx="11" cy="11" r="6" fill="none" stroke="currentColor" stroke-width="1.8" />
                  <line x1="11" y1="8" x2="11" y2="14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                  <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                </svg>
              </span>
            </button>
            <button
              class={styles.controlBtn}
              onClick={handleZoomOut}
              title="Zoom out"
              aria-label="Zoom out"
            >
              <span aria-hidden="true" class={styles.controlBtnIcon}>
                <svg
                  viewBox="0 0 24 24"
                  role="presentation"
                >
                  <circle cx="11" cy="11" r="6" fill="none" stroke="currentColor" stroke-width="1.8" />
                  <line x1="8" y1="11" x2="14" y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
                </svg>
              </span>
            </button>
            <button
              class={styles.controlBtn}
              onClick={() => applyTheme(theme() === 'dark' ? 'light' : 'dark')}
              title="Toggle light and dark mode"
              aria-label="Toggle light and dark mode"
            >
              <span aria-hidden="true" class={styles.controlBtnIcon}>
                <svg
                  viewBox="0 0 24 24"
                  role="presentation"
                >
                  <path
                    d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
                    fill="currentColor"
                  />
                </svg>
              </span>
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

      {/* Left-side panel containing Legend, Filters and Mining controls */}
      <div
        class={`${styles.leftPanel} ${isRightPanelOpen() ? styles.open : styles.collapsed}`}
        style={{ left: `${panelPos().x}px`, top: `${panelPos().y}px`, position: 'fixed' }}
      >
        {/* Drag handle - user can drag the panel by this bar */}
        <div
          class={styles.dragHandle}
          onPointerDown={(e) => startPanelDrag(e as unknown as PointerEvent)}
          title="Drag to reposition"
        />
        <button
          class={styles.panelToggleBtn}
          onClick={() => setIsRightPanelOpen(p => !p)}
          onPointerDown={(e) => startPanelDrag(e as unknown as PointerEvent)}
          aria-label={isRightPanelOpen() ? 'Close side panel' : 'Open side panel'}
          title={isRightPanelOpen() ? 'Hide panel' : 'Show panel'}
        >
          {/* Flip chevron direction for left-side panel */}
          {isRightPanelOpen() ? '❮' : '❯'}
        </button>

        <div class={styles.panelContent}>
          <EnhancedLegend
            graphData={graphData()}
            onFilterChange={handleFilterChange}
            filterState={filterState()}
          />
        </div>
      </div>

      {/* Floating bottom-center mining control (now using MiningInterface component) */}
      <div style={{ position: 'fixed', left: '50%', transform: 'translateX(-50%)', bottom: '28px', 'z-index': '1150' }}>
        <MiningInterface onMiningStart={startMiningUnified} onPatternsFound={handlePatternsFound} />
      </div>

      {/* Floating Chat Toggle Button */}
      <button
        class={styles.chatToggle}
        onClick={() => setIsChatOpen(prev => !prev)}
        aria-label="Toggle AI Assistant"
      >
        🤖
      </button>

      {/* Chat Interface - Opens when mining completes or user clicks the button */}
      <ChatInterface
        isOpen={isChatOpen()}
        onClose={() => setIsChatOpen(false)}
        conjunctSize={currentConjunctSize()}
        onVisualize={handleVisualize}
        miningResults={miningResults()}
        onMiningStart={startMiningUnified}
      />
    </div>
  );
};

export default App;
