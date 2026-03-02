import { createSignal, createEffect, Show, onCleanup } from 'solid-js';
import { Portal } from 'solid-js/web';
import ButtonParticleEffects from './ButtonEffects';
import './MiningInterface.css';

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
  const [conjunctionCount, setConjunctionCount] = createSignal(5);
  const [minSupport, setMinSupport] = createSignal(3);
  const [showResult, setShowResult] = createSignal(false);
  const [miningProgress, setMiningProgress] = createSignal(0);
  const [miningStatus, setMiningStatus] = createSignal('Preparing mines...');

  const statusMessages = [
    'Scanning AtomSpace...',
    'Digging for conjuncts...',
    'Filtering patterns...',
    'Extracting gold...',
    'Refining results...'
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

    const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

    try {
      // Start mining job
      const response = await fetch(`${API_BASE}/api/mine`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ conjunction_count: conjunctionCount(), min_support: minSupport() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const jobData = await response.json();

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
            <label for="conjunction-count" class="parameter-label">Conjunct count</label>
            <input
              id="conjunction-count"
              type="number"
              min="1"
              max="10"
              value={conjunctionCount()}
              onInput={(e) => setConjunctionCount(parseInt(e.target.value) || 5)}
              disabled={isMining()}
              class="separate-conj-input"
              aria-label="Conjunct count"
              title="Number of conditions joined in each mined pattern."
            />
            <span class="parameter-hint">Pattern complexity</span>
          </div>

          <div class="parameter-divider" />

          <div class="parameter-field">
            <label for="min-support" class="parameter-label">Min support</label>
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
            <span class="parameter-hint">Frequency threshold</span>
          </div>
        </div>

        <div class="button-wrapper" style={{ position: 'relative' }}>
          <ButtonParticleEffects active={!isMining()} />
          <button
            class={`mine-button ${isMining() ? 'mining' : ''}`}
            onClick={startMining}
            disabled={isMining()}
          >
            <div class="button-content generate-sparkles">
              <Show when={!isMining()}>
                <span class="button-text">Mine</span>
              </Show>
              <Show when={isMining()}>
                <div class="mining-animation">
                  <div class="progress-bar-container">
                    <div class="progress-bar-fill" style={{ width: `${miningProgress()}%` }}></div>
                  </div>
                  <div class="mining-status-content">
                    <div class="pickaxe-swing">⛏️</div>
                    <span class="mining-text">{miningStatus()}</span>
                  </div>
                  <div class="sparkles">
                    <span class="sparkle">✨</span>
                    <span class="sparkle">⭐</span>
                    <span class="sparkle">💫</span>
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
          >
            📜
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
                <h2>🏆 The Gold 🏆</h2>
                <p class="subtitle">Precious patterns extracted from the depths</p>
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
                <h2>⚠️ Mining Failed ⚠️</h2>
                <p class="subtitle">The mine collapsed!</p>
              </div>
              <div class="result-content">
                <div class="error-message">
                  <p>{miningResult()?.error}</p>
                </div>
              </div>
            </Show>

            <button class="close-button" onClick={closeResult}>
              ×
            </button>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default MiningInterface;