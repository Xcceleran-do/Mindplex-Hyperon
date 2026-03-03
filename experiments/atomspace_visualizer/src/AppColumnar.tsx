import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource, Show, For, onCleanup } from 'solid-js';

import ColumnarVisualizer from './components/ColumnarVisualizer/ColumnarVisualizer';
import EnhancedLegend from './components/Legend/EnhancedLegend';
import ChatInterface from './components/ChatInterface/ChatInterface';
import MiningInterface from './components/MiningInterface/MiningInterface';
import IngestionForm from './components/IngestionForm/IngestionForm';
import { API_CONFIG } from './config/api';
import { GraphData, GraphNode, FilterState } from './types';
import { MettaParserImpl } from './services/parser/MettaParser';
import { ColumnarTransformer } from './services/graph/ColumnarTransformer';

import './styles/variables.css';
import './styles/components.css';
import './styles/themes.css';
import styles from './AppColumnar.module.css';

const MAX_VISUALIZATION_ARTICLES = Number(import.meta.env.VITE_MAX_VIS_ARTICLES || 1500);

const limitTriplesForVisualization = (triples: ReturnType<MettaParserImpl['extractTriples']>) => {
  if (!Number.isFinite(MAX_VISUALIZATION_ARTICLES) || MAX_VISUALIZATION_ARTICLES <= 0) {
    return triples;
  }

  const allowedArticles = new Set<string>();
  const limited: typeof triples = [];

  for (const triple of triples) {
    const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
    const articleId = subjects[0];

    if (allowedArticles.has(articleId)) {
      limited.push(triple);
      continue;
    }

    if (allowedArticles.size < MAX_VISUALIZATION_ARTICLES) {
      allowedArticles.add(articleId);
      limited.push(triple);
    }
  }

  return limited;
};

const buildVisualizationSubset = (mettaText: string, maxArticles: number) => {
  if (!Number.isFinite(maxArticles) || maxArticles <= 0) {
    return mettaText;
  }

  const lines = mettaText.split('\n');
  const selectedLines: string[] = [];
  const allowedArticles = new Set<string>();

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith(';')) {
      continue;
    }

    const match = line.match(/^\(\s*(?:\(\s*)?[^\s()]+\s+([^\s()]+)\s+/);
    if (!match) {
      continue;
    }

    const articleId = match[1];
    if (!allowedArticles.has(articleId)) {
      if (allowedArticles.size >= maxArticles) {
        continue;
      }
      allowedArticles.add(articleId);
    }

    selectedLines.push(rawLine);
  }

  return selectedLines.join('\n');
};

const App: Component = () => {
  const [showVisualizer, setShowVisualizer] = createSignal(!API_CONFIG.INGESTION_ENABLED);

  // Load initial text from data.metta file
  const [initialTextResource, { refetch }] = createResource(async () => {
    try {
      // Add timestamp to prevent caching
      const response = await fetch(`/data.metta?t=${Date.now()}`);
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

  const handleIngestionComplete = () => {
    refetch();
    setShowVisualizer(true);
  };

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
  const [miningNotice, setMiningNotice] = createSignal<{ type: 'info' | 'success' | 'error'; text: string } | null>(null);
  const [isRulesModalOpen, setIsRulesModalOpen] = createSignal(false);

  // Zoom control state
  const [zoomTrigger, setZoomTrigger] = createSignal<{ action: 'in' | 'out' | 'recenter' | null; timestamp: number }>({ action: null, timestamp: 0 });

  // Animation state
  let animationInterval: number | undefined;

  // Initialize parser and columnar transformer
  const parser = new MettaParserImpl();
  const columnarTransformer = new ColumnarTransformer();

  // Parse and transform data to columnar format
  createEffect(() => {
    if (mettaText().trim()) {
      try {
        const visualizationText = buildVisualizationSubset(mettaText(), MAX_VISUALIZATION_ARTICLES);
        const triples = parser.extractTriples(visualizationText);
        const visualizationTriples = limitTriplesForVisualization(triples);
        const columnarData = columnarTransformer.transformToColumnar(visualizationTriples);
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
  const [isSidebarCollapsed, setIsSidebarCollapsed] = createSignal(false);

  const startResizing = (e: MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  };

  const stopResizing = () => {
    setIsResizing(false);
  };

  const resize = (e: MouseEvent) => {
    if (isResizing()) {
      const newWidth = Math.max(250, Math.min(600, e.clientX - 20)); // Adjust for left margin
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
    onCleanup(() => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    });
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

  // Keyboard shortcuts
  createEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Zoom In: Ctrl + Plus or Equals
      if (e.ctrlKey && (e.key === '+' || e.key === '=')) {
        e.preventDefault();
        handleZoomIn();
      }
      // Zoom Out: Ctrl + Minus
      if (e.ctrlKey && e.key === '-') {
        e.preventDefault();
        handleZoomOut();
      }
      // Recenter: 'r' key
      if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
        handleRecenter();
      }
      // Reset: Ctrl + Shift + R (Careful not to block browser reload unless desired, actually browser is Ctrl+R)
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'r') {
        e.preventDefault();
        handleReset();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    onCleanup(() => window.removeEventListener('keydown', handleKeyDown));
  });

  const handleFilterChange = (filter: FilterState) => {
    setFilterState(filter);
  };

  const handleZoomIn = () => {
    setZoomTrigger({ action: 'in', timestamp: Date.now() });
  };

  const handleZoomOut = () => {
    setZoomTrigger({ action: 'out', timestamp: Date.now() });
  };

  const handleRecenter = () => {
    setZoomTrigger({ action: 'recenter', timestamp: Date.now() });
  };

  const handleReset = () => {
    // Reset filters and view
    handleFilterChange({
      active: false,
      articleIds: [],
      propertyFilters: []
    });
    handleRecenter();
  };

  // Unified mining flow
  const startMiningUnified = async (conjunctSize: number, minSupport: number = 3) => {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

    try {
      startMiningAnimation();
      setMiningNotice(null);

      const resp = await fetch(`${API_BASE}/api/mine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conjunction_count: conjunctSize, min_support: minSupport })
      });

      const job = await resp.json();

      if (!resp.ok) {
        throw new Error(job?.message || `Mining API error ${resp.status}`);
      }

      stopMiningAnimation();
      setCurrentConjunctSize(conjunctSize);

      if (job.status === 'no_results') {
        setMiningResults([]);
        setMiningNotice({
          type: 'info',
          text: job.message || 'No patterns found. Try lowering MinSup or conjunction count.'
        });
        return;
      }

      const patterns = Array.isArray(job.result) ? job.result : [];
      setMiningResults(patterns);
      setMiningNotice({
        type: 'success',
        text: `Mining completed with ${patterns.length} pattern${patterns.length === 1 ? '' : 's'}.`
      });

    } catch (err) {
      console.error('startMiningUnified error', err);
      stopMiningAnimation();
      setMiningResults([]);
      setMiningNotice({
        type: 'error',
        text: err instanceof Error ? err.message : 'Mining failed. Please try again.'
      });
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
      // Regex to match (Property $Variable "Value") pattern
      // Handles variables with special chars (e.g. $x-1) and properties with hyphens
      const regex = /\(([^\s()]+)\s+\$[^\s()]+\s+("[^"]+")\)/g;
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
    <Show when={showVisualizer()} fallback={<Show when={API_CONFIG.INGESTION_ENABLED}><IngestionForm onComplete={handleIngestionComplete} /></Show>}>
      <div
        class={styles.app}
        style={{ "--sidebar-width": `${sidebarWidth()}px` } as any}
      >
        {/* Floating Header */}
        <header class={styles.header}>
          <div class={styles.logo}>
            <span style={{ "font-size": "1.2em" }}>🧠</span>
            Mindplex Hyperon
          </div>
          <div class={styles.headerControls}>
            <div class={styles.miningWrapper}>
              <MiningInterface
                onMiningStart={startMiningUnified}
                onPatternsFound={handlePatternsFound}
                onShowRules={miningResults().length > 0 ? () => setIsRulesModalOpen(true) : undefined}
              />
              <Show when={miningNotice()}>
                {(notice) => (
                  <div
                    class={`${styles.miningStatus} ${
                      notice().type === 'error'
                        ? styles.miningStatusError
                        : notice().type === 'success'
                          ? styles.miningStatusSuccess
                          : styles.miningStatusInfo
                    }`}
                  >
                    {notice().text}
                  </div>
                )}
              </Show>
            </div>
          </div>
        </header>

        {/* Toggle Button for Sidebar (Visible when collapsed or expanded) */}
        <button
          class={`${styles.sidebarToggle} ${isSidebarCollapsed() ? styles.collapsed : ''}`}
          onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed())}
          title={isSidebarCollapsed() ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isSidebarCollapsed() ? '➤' : '◀'}
        </button>

        {/* Floating Glass Sidebar */}
        <aside class={`${styles.sidebar} ${isSidebarCollapsed() ? styles.sidebarCollapsed : ''}`}>
          <div class={styles.sidebarContent}>
            <div class={styles.sidebarHeader}>
              <div class={styles.sidebarTitle}>Tools</div>
              <button class={styles.minimizeBtn} onClick={() => setIsSidebarCollapsed(true)}>_</button>
            </div>

            <div class={styles.sidebarSection}>
              <div class={styles.sectionHeader} onClick={() => setIsChatOpen(!isChatOpen())}>
                <div class={styles.sectionTitle}>AI Assistant</div>
                <span class={styles.collapseIcon}>{isChatOpen() ? '▼' : '▶'}</span>
              </div>
              <Show when={isChatOpen()}>
                <div class={styles.chatContainer}>
                  <ChatInterface
                    isOpen={true}
                    onClose={() => { }}
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
          {/* Resizer Handle inside the container */}
          <div class={styles.resizer} onMouseDown={startResizing} title="Drag to resize"></div>
        </aside>

        {/* Main Canvas Area */}
        <main class={styles.mainContent}>
          <div class={styles.graphContainer}>
            <ColumnarVisualizer
              graphData={graphData()}
              onNodeSelect={handleNodeSelect}
              filterState={filterState()}
              onFilterChange={handleFilterChange}
              zoomTrigger={zoomTrigger()}
            />
          </div>

          {/* Floating Action Buttons */}
          <div class={styles.canvasControls}>
            <button class={styles.controlBtn} onClick={handleZoomIn} title="Zoom In (Ctrl+ +)">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="11" y1="8" x2="11" y2="14" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button class={styles.controlBtn} onClick={handleZoomOut} title="Zoom Out (Ctrl+ -)">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button class={styles.controlBtn} onClick={handleRecenter} title="Recenter View">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
              </svg>
            </button>
            <button class={styles.controlBtn} onClick={handleReset} title="Reset All Filters & View">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
            </button>
          </div>
        </main>

        {/* Rules Modal - Preserved but styled via global CSS */}
        <Show when={isRulesModalOpen()}>
          <div class={styles.modalOverlay} onClick={() => setIsRulesModalOpen(false)}>
            <div class={styles.modalContent} onClick={(e) => e.stopPropagation()}>
              <div class={styles.modalHeader}>
                <h3>Mined Rules ({miningResults().length})</h3>
                <button class={styles.closeBtn} onClick={() => setIsRulesModalOpen(false)}>×</button>
              </div>
              <div class={styles.modalBody}>
                <Show when={miningResults().length > 0} fallback={<p>No rules mined yet.</p>}>
                  <div class={styles.rulesList}>
                    <For each={miningResults()}>
                      {(rule, index) => (
                        <div class={styles.ruleItem}>
                          <div class={styles.ruleHeader}>
                            <span class={styles.ruleIndex}>Rule {index() + 1}</span>
                            <span class={styles.ruleSupport}>Support: {rule.support}</span>
                          </div>
                          <pre class={styles.rulePattern}>{rule.pattern}</pre>
                          <button
                            class={styles.visualizeBtn}
                            onClick={() => {
                              handleVisualize(rule.pattern);
                              setIsRulesModalOpen(false);
                            }}
                          >
                            Visualize
                          </button>
                        </div>
                      )}
                    </For>
                  </div>
                </Show>
              </div>
            </div>
          </div>
        </Show>
      </div>
    </Show>
  );
};

export default App;
