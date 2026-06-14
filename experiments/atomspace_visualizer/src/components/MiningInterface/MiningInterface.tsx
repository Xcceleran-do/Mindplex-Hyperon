import { createSignal, createEffect, Show, onCleanup } from 'solid-js';
import { minePatterns } from '../../features/mining/api';
import { env } from '../../shared/config/env';
import './MiningInterface.css';

const MineIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 17.5 17.5 4M14.5 4H20v5.5M6.5 15.5l2 2M3.5 20.5l4-4" />
  </svg>
);

const RulesIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M7 4h10a2 2 0 0 1 2 2v14l-3-2-3 2-3-2-3 2V6a2 2 0 0 1 2-2Z" />
    <path d="M9 8h6M9 12h6M9 16h4" />
  </svg>
);

const CloseIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);

export interface MiningResult {
  jobId: string;
  status: 'running' | 'completed' | 'error';
  result?: any;
  error?: string;
  message?: string;
  duration?: number;
}

export interface MiningInterfaceProps {
  onMiningStart?: (conjunctSize: number, minSupport?: number) => void | Promise<void>;
  onMiningComplete?: (result: MiningResult) => void;
  onPatternsFound?: (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => void;
  onShowRules?: () => void;
}

const MiningInterface = (props: MiningInterfaceProps) => {
  const [isMining, setIsMining] = createSignal(false);
  const [miningResult, setMiningResult] = createSignal<MiningResult | null>(null);
  const maxConjunctionCount = Number.isFinite(env.maxConjunctionCount) && env.maxConjunctionCount > 0
    ? env.maxConjunctionCount
    : 10;
  const [conjunctionCount, setConjunctionCount] = createSignal(Math.min(5, maxConjunctionCount));
  const [minSupport, setMinSupport] = createSignal(3);
  const [showResult, setShowResult] = createSignal(false);
  const [miningProgress, setMiningProgress] = createSignal(0);
  const [miningStatus, setMiningStatus] = createSignal('Preparing inference...');

  const statusMessages = [
    'Scanning AtomSpace',
    'Compiling conjuncts',
    'Filtering support',
    'Ranking rules',
    'Normalizing results'
  ];

  createEffect(() => {
    if (isMining()) {
      const interval = setInterval(() => {
        setMiningProgress(prev => Math.min(prev + Math.random() * 5, 95));
        setMiningStatus(statusMessages[Math.floor(Math.random() * statusMessages.length)]);
      }, 800);
      onCleanup(() => clearInterval(interval));
    } else {
      setMiningProgress(0);
    }
  });

  const startMining = async () => {
    // Delegate to parent unified flow when available
    setIsMining(true);
    setMiningResult(null);
    setShowResult(false);
    if (props.onMiningStart) {
      try {
        await props.onMiningStart(conjunctionCount(), minSupport());
      } finally {
        setIsMining(false);
      }
      return;
    }

    // Fallback: if parent handler not provided, call API directly (legacy)

    try {
      const jobData = await minePatterns(conjunctionCount(), minSupport());

      // Mining now completes immediately with results
      setIsMining(false);

      const resultData = Array.isArray(jobData.result) ? jobData.result : [];
      const miningResult: MiningResult = {
        jobId: jobData.jobId,
        status: 'completed',
        result: resultData,
        message: jobData.message,
        duration: 0
      };
      setMiningResult(miningResult);
      setShowResult(true);

      if (jobData.status !== 'no_results' && resultData.length > 0) {
        console.log('MiningInterface (fallback): Calling onPatternsFound with conjunctSize:', conjunctionCount());
        props.onPatternsFound?.(resultData, conjunctionCount());
      }

    } catch (error) {
      console.error('Error starting mining:', error);
      setIsMining(false);
      setMiningResult({
        jobId: '',
        status: 'error',
        error: `Failed to start mining: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
      setShowResult(true);
    }
  };

  const closeResult = () => {
    setShowResult(false);
    setMiningResult(null);
  };

  // Drag state for the result card
  const [dragging, setDragging] = createSignal(false);
  const [dragPos, setDragPos] = createSignal({ x: 0, y: 0 });
  const [moved, setMoved] = createSignal(false);
  const dragOffset = { x: 0, y: 0 };

  const startDrag = (e: PointerEvent, el: HTMLElement) => {
    e.preventDefault();
    setDragging(true);
    setMoved(true);
    const rect = el.getBoundingClientRect();
    dragOffset.x = e.clientX - rect.left;
    dragOffset.y = e.clientY - rect.top;
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', endDrag);
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!dragging()) return;
    setDragPos({ x: e.clientX - dragOffset.x, y: e.clientY - dragOffset.y });
  };

  const endDrag = (_e: PointerEvent) => {
    setDragging(false);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', endDrag);
  };

  return (
    <div class="mining-interface">
      {/* Mining Control Panel */}
      <div class="mining-controls">
        <div class="parameter-panel">
          <div class="parameter-field">
            <label for="conjunction-count" class="parameter-label">Conditions / rule</label>
            <input
              id="conjunction-count"
              type="number"
              min="1"
              max={maxConjunctionCount}
              value={conjunctionCount()}
              onInput={(e) => setConjunctionCount(Math.min(maxConjunctionCount, Math.max(1, parseInt(e.target.value) || 5)))}
              disabled={isMining()}
              class="separate-conj-input"
              aria-label="Conjunct count"
              title="Number of conditions joined in each mined pattern."
            />
            <span class="parameter-hint">Complexity, max {maxConjunctionCount}</span>
          </div>

          <div class="parameter-divider" />

          <div class="parameter-field">
            <label for="min-support" class="parameter-label">Min examples</label>
            <input
              id="min-support"
              type="number"
              min="1"
              value={minSupport()}
              onInput={(e) => setMinSupport(parseInt(e.target.value) || 3)}
              disabled={isMining()}
              class="separate-conj-input"
              aria-label="Minimum support"
              title="Minimum number of occurrences required for a pattern to be returned."
            />
            <span class="parameter-hint">Required matches</span>
          </div>
        </div>

        <div class="button-wrapper">
          <button
            class={`mine-button ${isMining() ? 'mining' : ''}`}
            onClick={startMining}
            disabled={isMining()}
          >
            <div class="button-content">
              <Show when={!isMining()}>
                <MineIcon />
                <span class="button-text">Mine Rules</span>
              </Show>
              <Show when={isMining()}>
                <div class="mining-animation">
                  <div class="progress-bar-container">
                    <div class="progress-bar-fill" style={{ width: `${miningProgress()}%` }}></div>
                  </div>
                  <div class="mining-status-content">
                    <span class="button-spinner" />
                    <span class="mining-text">{miningStatus()}</span>
                  </div>
                </div>
              </Show>
            </div>
          </button>
        </div>

        <Show when={props.onShowRules}>
          <button
            class="show-rules-btn"
            onClick={props.onShowRules}
            title="Show Mined Rules"
            aria-label="Show mined rules"
          >
            <RulesIcon />
          </button>
        </Show>
      </div>



      {/* Result Card */}
      <Show when={showResult() && miningResult()}>
        <div class="result-overlay" onClick={closeResult}>
          <div
            class={`result-card ${dragging() ? 'dragging' : ''}`}
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => startDrag(e as unknown as PointerEvent, e.currentTarget as HTMLElement)}
            style={(() => {
              if (!moved()) return {};
              const p = dragPos();
              return { left: `${p.x}px`, top: `${p.y}px`, transform: 'none', bottom: 'auto' } as any;
            })()}
          >
            <Show when={miningResult()?.status === 'completed'}>
              <div class="result-header gold">
                <h2>Mining Results</h2>
                <p class="subtitle">Patterns returned by the PeTTa pipeline</p>
              </div>
              <div class="result-content">
                <Show when={Array.isArray(miningResult()?.result) && miningResult()?.result?.length === 0}>
                  <div class="error-message">
                    <p>{miningResult()?.message || 'No patterns found for the current parameters.'}</p>
                  </div>
                </Show>
                <div class="patterns-display">
                  <pre class="patterns-text">{JSON.stringify(miningResult()?.result, null, 2)}</pre>
                </div>
                <div class="mining-stats">
                  <p><strong>Duration:</strong> {miningResult()?.duration?.toFixed(2)}s</p>
                  <p><strong>Conjunctions:</strong> {conjunctionCount()}</p>
                </div>
              </div>
            </Show>

            <Show when={miningResult()?.status === 'error'}>
              <div class="result-header error">
                <h2>Mining Failed</h2>
                <p class="subtitle">The pipeline returned an error</p>
              </div>
              <div class="result-content">
                <div class="error-message">
                  <p>{miningResult()?.error}</p>
                </div>
              </div>
            </Show>

            <button class="close-button" onClick={closeResult} aria-label="Close mining result">
              <CloseIcon />
            </button>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default MiningInterface;
