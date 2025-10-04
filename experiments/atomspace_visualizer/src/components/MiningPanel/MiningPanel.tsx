// Mining Interface for pattern discovery
import { Component, createSignal, Show, For } from 'solid-js';
import { FilterState } from '../../types';
import styles from './MiningPanel.module.css';

export interface MiningResult {
  support: Array<{
    pattern: string;
    support: number;
    properties: Array<{
      property: string;
      value: string;
    }>;
  }>;
}

export interface MiningPanelProps {
  onFilterChange: (filter: FilterState) => void;
}

const MiningPanel: Component<MiningPanelProps> = (props) => {
  const [conjunctSize, setConjunctSize] = createSignal(2);
  const [isMining, setIsMining] = createSignal(false);
  const [miningResult, setMiningResult] = createSignal<MiningResult | null>(null);
  const [isResultCollapsed, setIsResultCollapsed] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [animationArticleIndex, setAnimationArticleIndex] = createSignal(0);

  // Animation interval for loading state
  let animationInterval: number | null = null;

  const startLoadingAnimation = () => {
    setAnimationArticleIndex(0);
    
    animationInterval = setInterval(() => {
      setAnimationArticleIndex(prev => (prev + 1) % 10); // Cycle through 10 articles
      
      // Highlight current article during animation
      props.onFilterChange({
        active: true,
        articleIds: [animationArticleIndex().toString()]
      });
    }, 1500); // 1.5 second intervals
  };

  const stopLoadingAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }
    // Clear filter
    props.onFilterChange({ active: false });
  };

  const startMining = async () => {
    setIsMining(true);
    setError(null);
    startLoadingAnimation();

    try {
      const response = await fetch('http://localhost:5000/api/mine', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conjunction_count: conjunctSize()
        }),
      });

      if (!response.ok) {
        throw new Error('Mining request failed');
      }

      const result = await response.json();
      
      // Parse the result
      const parsedResult = parseMiningResult(result);
      setMiningResult(parsedResult);
      setIsResultCollapsed(false);
    } catch (err) {
      console.error('Mining error:', err);
      setError('Failed to mine patterns. Make sure the mining API is running on port 5000.');
    } finally {
      setIsMining(false);
      stopLoadingAnimation();
    }
  };

  const parseMiningResult = (rawResult: any): MiningResult => {
    // Parse the mining result format
    // Expected format: [(supportOf (, (property $V0 value) ...) count)]
    const patterns: MiningResult['support'] = [];

    try {
      // Extract patterns from the result
      const resultStr = JSON.stringify(rawResult);
      const supportRegex = /supportOf\s*\([^)]+\)\s*(\d+)/g;
      let match;

      while ((match = supportRegex.exec(resultStr)) !== null) {
        const fullMatch = match[0];
        const support = parseInt(match[1]);
        
        // Extract properties and values
        const propertyRegex = /\((\w+)\s+\$V\d+\s+"([^"]+)"\)/g;
        const properties = [];
        let propMatch;
        
        while ((propMatch = propertyRegex.exec(fullMatch)) !== null) {
          properties.push({
            property: propMatch[1],
            value: propMatch[2]
          });
        }

        if (properties.length > 0) {
          patterns.push({
            pattern: fullMatch,
            support,
            properties
          });
        }
      }
    } catch (err) {
      console.error('Error parsing mining result:', err);
    }

    return { support: patterns };
  };

  const visualizePattern = (pattern: MiningResult['support'][0]) => {
    // Extract all property values to highlight
    const propertyFilters: Array<{ property: string; value: string }> = [];
    
    for (const prop of pattern.properties) {
      propertyFilters.push({
        property: prop.property,
        value: prop.value
      });
    }

    // Apply multi-select filter
    props.onFilterChange({
      active: true,
      propertyFilters: propertyFilters
    });

    // Collapse result card after visualization
    setIsResultCollapsed(true);
  };

  const clearResults = () => {
    setMiningResult(null);
    setIsResultCollapsed(true);
    props.onFilterChange({ active: false });
  };

  return (
    <div class={styles.miningPanel}>
      <div class={styles.miningControls}>
        <div class={styles.inputGroup}>
          <label for="conjunct-size">Conjunct Size:</label>
          <input
            id="conjunct-size"
            type="number"
            min="2"
            max="10"
            value={conjunctSize()}
            onInput={(e) => setConjunctSize(parseInt(e.currentTarget.value) || 2)}
            disabled={isMining()}
            class={styles.numberInput}
          />
        </div>
        <button
          onClick={startMining}
          disabled={isMining()}
          class={`${styles.mineButton} ${isMining() ? styles.mining : ''}`}
        >
          {isMining() ? (
            <>
              <span class={styles.spinner}></span>
              Mining...
            </>
          ) : (
            '⛏️ Mine Patterns'
          )}
        </button>
      </div>

      <Show when={error()}>
        <div class={styles.error}>
          {error()}
        </div>
      </Show>

      <Show when={isMining()}>
        <div class={styles.loadingAnimation}>
          <div class={styles.animationText}>
            Analyzing article {animationArticleIndex()}...
          </div>
          <div class={styles.progressBar}>
            <div class={styles.progressFill}></div>
          </div>
        </div>
      </Show>

      <Show when={miningResult() && miningResult()!.support.length > 0}>
        <div class={styles.resultsCard}>
          <div 
            class={styles.resultsHeader}
            onClick={() => setIsResultCollapsed(!isResultCollapsed())}
          >
            <h4>Mining Results ({miningResult()!.support.length} patterns)</h4>
            <button class={styles.collapseButton}>
              {isResultCollapsed() ? '▼' : '▲'}
            </button>
          </div>
          
          <Show when={!isResultCollapsed()}>
            <div class={styles.resultsContent}>
              <For each={miningResult()!.support}>
                {(pattern, index) => (
                  <div class={styles.patternCard}>
                    <div class={styles.patternHeader}>
                      <span class={styles.patternIndex}>Pattern {index() + 1}</span>
                      <span class={styles.supportBadge}>Support: {pattern.support}</span>
                    </div>
                    <div class={styles.patternProperties}>
                      <For each={pattern.properties}>
                        {(prop) => (
                          <div class={styles.propertyTag}>
                            <span class={styles.propertyName}>{prop.property}:</span>
                            <span class={styles.propertyValue}>{prop.value}</span>
                          </div>
                        )}
                      </For>
                    </div>
                    <button
                      onClick={() => visualizePattern(pattern)}
                      class={styles.visualizeButton}
                    >
                      👁️ Visualize
                    </button>
                  </div>
                )}
              </For>
              <button onClick={clearResults} class={styles.clearButton}>
                Clear Results
              </button>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
};

export default MiningPanel;
