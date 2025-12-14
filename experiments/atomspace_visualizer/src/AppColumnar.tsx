import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource, Show } from 'solid-js';

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
  const [isChatOpen, setIsChatOpen] = createSignal(true); // Default to open since it's in sidebar
  const [theme, setTheme] = createSignal<string>(localStorage.getItem('theme') || 'auto');
  
  // Sidebar resizing
  const [sidebarWidth, setSidebarWidth] = createSignal(320);
  const [isResizing, setIsResizing] = createSignal(false);

  const startResizing = (e: MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  };

  const stopResizing = () => {
    setIsResizing(false);
  };

  const resize = (e: MouseEvent) => {
    if (isResizing()) {
      const newWidth = Math.max(250, Math.min(600, e.clientX));
      setSidebarWidth(newWidth);
    }
  };

  createEffect(() => {
    if (isResizing()) {
      window.addEventListener('mousemove', resize);
      window.addEventListener('mouseup', stopResizing);
    } else {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    }
  });

  const applyTheme = (t: string) => {
    if (t === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else if (t === 'light') {
      document.documentElement.removeAttribute('data-theme');
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

  // Unified mining flow
  const startMiningUnified = async (conjunctSize: number) => {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

    try {
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

      stopMiningAnimation();
      setMiningResults(job.result || []);
      setCurrentConjunctSize(conjunctSize);

    } catch (err) {
      console.error('startMiningUnified error', err);
      stopMiningAnimation();
    }
  };

  const handlePatternsFound = (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => {
    stopMiningAnimation();
    setMiningResults(patterns);
    if (conjunctSize) {
      setCurrentConjunctSize(conjunctSize);
    }
  };

  const startMiningAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
    }
    
    const articles: string[] = [];
    for (const node of graphData().nodes) {
      if (node.metadata.columnType === 'article') {
        articles.push(node.metadata.originalExpression || node.label);
      }
    }
    
    if (articles.length === 0) return;
    
    let currentIndex = 0;
    const intervalTime = 1000;
    
    animationInterval = setInterval(() => {
      currentIndex = currentIndex % articles.length;
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
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = undefined;
    }
    handleFilterChange({
      active: false,
      articleIds: [],
      propertyFilters: []
    });
  };

  const handleVisualize = (filterState: FilterState | string) => {
    if (typeof filterState === 'string') {
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
      handleFilterChange(filterState);
    }
  };

  return (
    <div 
      class={styles.app}
      style={{ "--sidebar-width": `${sidebarWidth()}px` } as any}
    >
      {/* Header */}
      <header class={styles.header}>
        <div class={styles.logo}>
          <span style={{ "font-size": "1.5em" }}>🧠</span>
          Mindplex Hyperon
        </div>
        <div class={styles.headerControls}>
          <div class={styles.miningWrapper}>
            <MiningInterface onMiningStart={startMiningUnified} onPatternsFound={handlePatternsFound} />
          </div>
        </div>
      </header>

      {/* Sidebar */}
      <aside class={styles.sidebar}>
        <div class={styles.sidebarContent}>
          <div class={styles.sidebarSection}>
            <div class={styles.sectionHeader} onClick={() => setIsChatOpen(!isChatOpen())}>
              <div class={styles.sectionTitle}>AI Assistant</div>
              <span class={styles.collapseIcon}>{isChatOpen() ? '▼' : '▶'}</span>
            </div>
            <Show when={isChatOpen()}>
              <div class={styles.chatContainer}>
                <ChatInterface
                  isOpen={true}
                  onClose={() => {}}
                  conjunctSize={currentConjunctSize()}
                  onVisualize={handleVisualize}
                  miningResults={miningResults()}
                  onMiningStart={startMiningUnified}
                />
              </div>
            </Show>
          </div>

          <div class={styles.sidebarSection}>
            <div class={styles.sectionTitle}>Legend</div>
            <EnhancedLegend
              graphData={graphData()}
              onFilterChange={handleFilterChange}
              filterState={filterState()}
            />
          </div>
          
          <div 
            class={styles.themeToggle} 
            onClick={() => applyTheme(theme() === 'dark' ? 'light' : 'dark')}
          >
            <Show when={theme() === 'dark'} fallback={<span>🌙 Dark Mode</span>}>
              <span>☀️ Light Mode</span>
            </Show>
          </div>
        </div>
        <div class={styles.resizer} onMouseDown={startResizing}></div>
      </aside>

      {/* Main Content */}
      <main class={styles.mainContent}>
        <div class={styles.graphContainer}>
          <ColumnarVisualizer
            graphData={graphData()}
            onNodeSelect={handleNodeSelect}
            filterState={filterState()}
            onFilterChange={handleFilterChange}
          />
        </div>

        {/* Canvas Controls */}
        <div class={styles.canvasControls}>
          <button class={styles.controlBtn} onClick={handleZoomIn} title="Zoom In">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
              <line x1="11" y1="8" x2="11" y2="14" />
              <line x1="8" y1="11" x2="14" y2="11" />
            </svg>
          </button>
          <button class={styles.controlBtn} onClick={handleZoomOut} title="Zoom Out">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
              <line x1="8" y1="11" x2="14" y2="11" />
            </svg>
          </button>
        </div>
      </main>
    </div>
  );
};

export default App;
