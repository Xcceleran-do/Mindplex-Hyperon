import { createSignal, Show } from 'solid-js';
import { DEFAULT_CONJUNCTION_COUNT, DEFAULT_MIN_SUPPORT } from '../../features/mining/defaults';
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

export interface MiningInterfaceProps {
  onMiningStart: (conjunctSize: number, minSupport?: number) => void | Promise<void>;
  onShowRules?: () => void;
}

const MiningInterface = (props: MiningInterfaceProps) => {
  const [isMining, setIsMining] = createSignal(false);
  const [conjunctionCount, setConjunctionCount] = createSignal(DEFAULT_CONJUNCTION_COUNT);
  const [minSupport, setMinSupport] = createSignal(DEFAULT_MIN_SUPPORT);

  const startMining = async () => {
    setIsMining(true);
    try {
      await props.onMiningStart(conjunctionCount(), minSupport());
    } finally {
      setIsMining(false);
    }
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
              value={conjunctionCount()}
              onInput={(e) => setConjunctionCount(Math.max(1, parseInt(e.target.value) || DEFAULT_CONJUNCTION_COUNT))}
              disabled={isMining()}
              class="separate-conj-input"
              aria-label="Conjunct count"
              title="Number of conditions joined in each mined pattern."
            />
            <span class="parameter-hint">Required conditions</span>
          </div>

          <div class="parameter-divider" />

          <div class="parameter-field">
            <label for="min-support" class="parameter-label">Min examples</label>
            <input
              id="min-support"
              type="number"
              min="1"
              value={minSupport()}
              onInput={(e) => setMinSupport(Math.max(1, parseInt(e.target.value) || DEFAULT_MIN_SUPPORT))}
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
                    <div class="progress-bar-fill"></div>
                  </div>
                  <div class="mining-status-content">
                    <span class="button-spinner" />
                    <span class="mining-text">Mining rules...</span>
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
    </div>
  );
};

export default MiningInterface;
