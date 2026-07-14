import type { Component } from 'solid-js';
import { createSignal, createEffect, createResource, Show, For, onCleanup, createMemo } from 'solid-js';

import EnhancedLegend from './components/Legend/EnhancedLegend';
import ChatInterface from './components/ChatInterface/ChatInterface';
import MiningInterface from './components/MiningInterface/MiningInterface';
import IngestionForm from './components/IngestionForm/IngestionForm';
import SimulatorPanel from './components/SimulatorPanel/SimulatorPanel';
import { loadMettaDataset } from './features/mettaData/api';
import { useResizableSidebar } from './features/layout/useResizableSidebar';
import { useMiningWorkflow } from './features/mining/model/useMiningWorkflow';
import { filterStateFromPattern } from './features/patterns/patternFilters';
import { useTheme } from './features/theme/useTheme';
import SemanticAtlas from './features/visualization/atlas/SemanticAtlas';
import { useVisualizationData } from './features/visualization/model/useVisualizationData';
import { env } from './shared/config/env';
import { GraphNode, FilterState } from './types';

import './styles/variables.css';
import './styles/components.css';
import './styles/themes.css';
import styles from './AppColumnar.module.css';

const BrandGlyph = () => (
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <path d="M16 3.5 27 9.9v12.2l-11 6.4-11-6.4V9.9L16 3.5Z" />
    <path d="M10.8 13.1h10.4M10.8 18.9h10.4M16 8.9v14.2" />
    <circle cx="16" cy="16" r="2.2" />
  </svg>
);

const ChevronIcon = (props: { direction: 'left' | 'right' | 'up' | 'down' }) => {
  const rotation = {
    right: 0,
    down: 90,
    left: 180,
    up: 270,
  }[props.direction];

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" style={{ transform: `rotate(${rotation}deg)` }}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
};

const ThemeGlyph = (props: { mode: 'dark' | 'light' }) => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <Show
      when={props.mode === 'light'}
      fallback={<path d="M20.4 14.8A7.5 7.5 0 0 1 9.2 3.6 8.8 8.8 0 1 0 20.4 14.8Z" />}
    >
      <path d="M12 4v2.2M12 17.8V20M4 12h2.2M17.8 12H20M6.3 6.3l1.6 1.6M16.1 16.1l1.6 1.6M17.7 6.3l-1.6 1.6M7.9 16.1l-1.6 1.6" />
      <circle cx="12" cy="12" r="3.6" />
    </Show>
  </svg>
);

const CloseIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);

type SidebarTool = 'ask' | 'simulate' | 'explore';

const App: Component = () => {
  const [showVisualizer, setShowVisualizer] = createSignal(!env.ingestionEnabled);

  // Load data.metta only after ingestion completes, unless ingestion is
  // disabled and the existing dataset should be opened directly.
  const [initialTextResource] = createResource(showVisualizer, (enabled) =>
    enabled ? loadMettaDataset() : Promise.resolve('')
  );

  const handleIngestionComplete = () => {
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

  const graphData = useVisualizationData(mettaText);
  const suggestedArticleId = createMemo(() => {
    const article = graphData().nodes.find((node) => node.metadata.columnType === 'article');
    return article?.metadata.originalExpression || article?.label;
  });

  const [filterState, setFilterState] = createSignal<FilterState>({
    active: false,
    articleIds: [],
    propertyFilters: []
  });
  const handleFilterChange = (filter: FilterState) => {
    setFilterState(filter);
  };

  const [isRulesModalOpen, setIsRulesModalOpen] = createSignal(false);
  const [expandedRules, setExpandedRules] = createSignal<Record<number, boolean>>({});

  // Zoom control state
  const [zoomTrigger, setZoomTrigger] = createSignal<{ action: 'in' | 'out' | 'recenter' | null; timestamp: number }>({ action: null, timestamp: 0 });

  // Event handlers
  const handleNodeSelect = (node: GraphNode) => {
    console.log('Selected node:', node.label);
  };

  const [activeSidebarTool, setActiveSidebarTool] = createSignal<SidebarTool>('ask');
  const { theme, applyTheme } = useTheme();
  const { sidebarWidth, isSidebarCollapsed, setIsSidebarCollapsed, startResizing } = useResizableSidebar();
  const {
    miningResults,
    currentConjunctSize,
    miningNotice,
    startMining,
    setPatternsFound
  } = useMiningWorkflow(graphData, handleFilterChange);
  const workspaceStats = createMemo(() => {
    const data = graphData();
    const articles = data.nodes.filter((node) => node.metadata.columnType === 'article').length;
    const properties = new Set(
      data.nodes
        .filter((node) => node.metadata.columnType === 'header')
        .map((node) => node.metadata.originalExpression || node.label)
        .filter(Boolean)
    ).size;
    return {
      articles,
      properties,
      rules: miningResults().length,
    };
  });
  let lastRulesPanelSignature = '';

  // Open chat and the top-right rules panel automatically when fresh mining
  // results arrive, regardless of whether mining started from the button or
  // the chat command path.
  createEffect(() => {
    const results = miningResults();
    if (results && results.length > 0) {
      setActiveSidebarTool('ask');
      const signature = `${currentConjunctSize() ?? 'unknown'}:${results
        .map((result) => `${result.pattern}:${result.support}`)
        .join('|')}`;
      if (signature !== lastRulesPanelSignature) {
        lastRulesPanelSignature = signature;
        setIsRulesModalOpen(true);
      }
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

  const handlePatternsFound = (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => {
    setPatternsFound(patterns, conjunctSize);
  };

  const handleVisualize = (filterState: FilterState | string) => {
    if (typeof filterState === 'string') {
      const nextFilter = filterStateFromPattern(filterState);
      if (nextFilter) {
        handleFilterChange(nextFilter);
      }
    } else {
      handleFilterChange(filterState);
    }
  };

  const toggleRule = (index: number) => {
    setExpandedRules((current) => ({
      ...current,
      [index]: !current[index],
    }));
  };

  return (
    <Show when={showVisualizer()} fallback={<Show when={env.ingestionEnabled}><IngestionForm onComplete={handleIngestionComplete} /></Show>}>
      <div
        class={styles.app}
        style={{ "--sidebar-width": `${sidebarWidth()}px` } as any}
      >
        {/* Floating Header */}
        <header class={styles.header}>
          <div class={styles.logo}>
            <span class={styles.logoMark}><BrandGlyph /></span>
            <span class={styles.logoCopy}>
              <span class={styles.logoText}>Mindplex Hyperon</span>
              <span class={styles.logoSubtext}>Article reasoning workbench</span>
            </span>
          </div>
          <div class={styles.workspaceStats} aria-label="Workspace statistics">
            <span><strong>{workspaceStats().articles}</strong> articles</span>
            <span><strong>{workspaceStats().properties}</strong> attributes</span>
            <span><strong>{workspaceStats().rules}</strong> rules</span>
          </div>
          <div class={styles.headerControls}>
            <div class={styles.miningWrapper}>
              <MiningInterface
                onMiningStart={startMining}
                onShowRules={miningResults().length > 0 ? () => setIsRulesModalOpen(true) : undefined}
              />
              <Show when={miningNotice()}>
                {(notice) => (
                  <div
                    class={`${styles.miningStatus} ${
                      notice().type === 'error'
                        ? styles.miningStatusError
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

        {/* External toggle is only interactive while the sidebar is collapsed. */}
        <button
          class={`${styles.sidebarToggle} ${isSidebarCollapsed() ? styles.collapsed : ''}`}
          onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed())}
          title={isSidebarCollapsed() ? "Expand Sidebar" : "Collapse Sidebar"}
          aria-label={isSidebarCollapsed() ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          <ChevronIcon direction={isSidebarCollapsed() ? 'right' : 'left'} />
        </button>

        {/* Floating Glass Sidebar */}
        <aside class={`${styles.sidebar} ${isSidebarCollapsed() ? styles.sidebarCollapsed : ''}`}>
          <div class={styles.sidebarContent}>
            <div class={styles.sidebarHeader}>
              <div>
                <div class={styles.sidebarTitle}>Workspace</div>
                <div class={styles.sidebarSubtitle}>Reason over your article portfolio.</div>
              </div>
              <button class={styles.minimizeBtn} onClick={() => setIsSidebarCollapsed(true)} aria-label="Collapse tools">
                <ChevronIcon direction="left" />
              </button>
            </div>

            <div class={styles.sidebarTabs} role="tablist" aria-label="Reasoning tools">
              <button
                type="button"
                role="tab"
                aria-selected={activeSidebarTool() === 'ask'}
                class={`${styles.sidebarTab} ${activeSidebarTool() === 'ask' ? styles.sidebarTabActive : ''}`}
                onClick={() => setActiveSidebarTool('ask')}
              >
                Ask
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeSidebarTool() === 'simulate'}
                class={`${styles.sidebarTab} ${activeSidebarTool() === 'simulate' ? styles.sidebarTabActive : ''}`}
                onClick={() => setActiveSidebarTool('simulate')}
              >
                Simulate
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeSidebarTool() === 'explore'}
                class={`${styles.sidebarTab} ${activeSidebarTool() === 'explore' ? styles.sidebarTabActive : ''}`}
                onClick={() => setActiveSidebarTool('explore')}
              >
                Explore
              </button>
            </div>

            <div class={styles.sidebarBody}>
              <div
                role="tabpanel"
                class={styles.sidebarToolPanel}
                classList={{ [styles.sidebarToolPanelHidden]: activeSidebarTool() !== 'simulate' }}
              >
                <div class={styles.toolIntro}>
                  <div class={styles.sectionTitle}>What-if simulator</div>
                  <div class={styles.sectionHint}>Test a hypothetical article against mined rules.</div>
                </div>
                <SimulatorPanel graphData={graphData()} minedRuleCount={miningResults().length} />
              </div>

              <div
                role="tabpanel"
                class={styles.sidebarToolPanel}
                classList={{ [styles.sidebarToolPanelHidden]: activeSidebarTool() !== 'ask' }}
              >
                <div class={styles.chatContainer}>
                  <ChatInterface
                    isOpen={true}
                    onClose={() => { }}
                    conjunctSize={currentConjunctSize()}
                    onVisualize={handleVisualize}
                    miningResults={miningResults()}
                    onMiningStart={startMining}
                    onPatternsFound={handlePatternsFound}
                    onShowRules={() => setIsRulesModalOpen(true)}
                    suggestedArticleId={suggestedArticleId()}
                  />
                </div>
              </div>

              <div
                role="tabpanel"
                class={styles.sidebarToolPanel}
                classList={{ [styles.sidebarToolPanelHidden]: activeSidebarTool() !== 'explore' }}
              >
                <div class={styles.toolIntro}>
                  <div class={styles.sectionTitle}>Atlas filters</div>
                  <div class={styles.sectionHint}>Select values to narrow and compare the graph.</div>
                </div>
                <EnhancedLegend
                  graphData={graphData()}
                  onFilterChange={handleFilterChange}
                  filterState={filterState()}
                />
              </div>
            </div>

            <div
              class={styles.themeToggle}
              onClick={() => applyTheme(theme() === 'dark' ? 'light' : 'dark')}
            >
              <Show when={theme() === 'dark'} fallback={<><ThemeGlyph mode="dark" /><span>Dark</span></>}>
                <ThemeGlyph mode="light" /><span>Light</span>
              </Show>
            </div>
          </div>
          {/* Resizer Handle inside the container */}
          <div class={styles.resizer} onMouseDown={startResizing} title="Drag to resize"></div>
        </aside>

        {/* Main Canvas Area */}
        <main class={styles.mainContent}>
          <div class={styles.graphContainer}>
            <SemanticAtlas
              graphData={graphData()}
              onNodeSelect={handleNodeSelect}
              filterState={filterState()}
              onFilterChange={handleFilterChange}
              zoomTrigger={zoomTrigger()}
              showOverview={!isRulesModalOpen()}
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
                <button class={styles.closeBtn} onClick={() => setIsRulesModalOpen(false)} aria-label="Close rules modal">
                  <CloseIcon />
                </button>
              </div>
              <div class={styles.modalBody}>
                <Show when={miningResults().length > 0} fallback={<p>No rules mined yet.</p>}>
                  <div class={styles.rulesList}>
                    <For each={miningResults()}>
                      {(rule, index) => (
                        <div class={styles.ruleItem}>
                          <button
                            type="button"
                            class={styles.ruleHeader}
                            onClick={() => toggleRule(index())}
                            aria-expanded={Boolean(expandedRules()[index()])}
                          >
                            <span class={styles.ruleHeaderMeta}>
                              <span class={styles.ruleIndex}>Rule {index() + 1}</span>
                              <span class={styles.ruleSupport}>Support: {rule.support}</span>
                            </span>
                            <span class={styles.ruleChevron}>
                              <ChevronIcon direction={expandedRules()[index()] ? 'up' : 'down'} />
                            </span>
                          </button>
                          <Show when={expandedRules()[index()]}>
                            <div class={styles.ruleBody}>
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
                          </Show>
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
