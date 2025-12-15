import { Component, createSignal, createMemo, For, Show } from 'solid-js';
import { GraphData } from '../../types';
import styles from './SimulationMode.module.css';

interface SimulationModeProps {
  graphData: GraphData;
  onClose: () => void;
}

const SimulationMode: Component<SimulationModeProps> = (props) => {
  const [selectedValues, setSelectedValues] = createSignal<Record<string, string>>({});

  // Extract available properties and their values from graph data
  const availableProperties = createMemo(() => {
    const propsMap = new Map<string, Set<string>>();
    
    props.graphData.nodes.forEach(node => {
      if (node.metadata.columnType === 'property' && node.metadata.propertyName) {
        const propName = node.metadata.propertyName;
        if (!propsMap.has(propName)) {
          propsMap.set(propName, new Set());
        }
        if (node.label !== 'None') {
          propsMap.get(propName)!.add(node.label);
        }
      }
    });

    // Convert to array and sort
    return Array.from(propsMap.entries())
      .filter(([name]) => !['engagement', 'audience-expertise'].includes(name.toLowerCase()))
      .map(([name, values]) => ({
        name,
        values: Array.from(values).sort()
      })).sort((a, b) => a.name.localeCompare(b.name));
  });

  const handleValueChange = (property: string, value: string) => {
    setSelectedValues(prev => {
      const next = { ...prev };
      if (value === '') {
        delete next[property];
      } else {
        next[property] = value;
      }
      return next;
    });
  };

  // Mock calculation logic
  const calculatedProbabilities = createMemo(() => {
    const selections = selectedValues();
    const seed = Object.entries(selections)
      .reduce((acc, [k, v]) => acc + k.length + v.length, 0);
    
    // Deterministic pseudo-random based on selection length
    const pseudoRandom = (offset: number) => {
      const x = Math.sin(seed + offset) * 10000;
      return x - Math.floor(x);
    };

    const generateDistribution = (offsetBase: number) => {
      let high = 0.2 + (pseudoRandom(offsetBase) * 0.6);
      let med = 0.1 + (pseudoRandom(offsetBase + 1) * 0.5);
      let low = 1 - (high + med);
      
      // Normalize
      const total = high + med + low;
      return {
        High: Math.round((high / total) * 100),
        Medium: Math.round((med / total) * 100),
        Low: Math.round((low / total) * 100)
      };
    };

    // Generate distinct distributions for each audience type
    // If selections are empty, return balanced defaults
    if (Object.keys(selections).length === 0) {
      const balanced = { High: 33, Medium: 34, Low: 33 };
      return {
        Expert: balanced,
        Intermediate: balanced,
        Novice: balanced
      };
    }

    return {
      Expert: generateDistribution(1),
      Intermediate: generateDistribution(10),
      Novice: generateDistribution(20)
    };
  });

  const formatPropName = (name: string) => 
    name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

  return (
    <div class={styles.container}>
      <div class={styles.header}>
        <h2>Metadata Simulation Playground</h2>
        <p>Adjust metadata parameters to simulate and predict audience engagement and expertise levels.</p>
      </div>

      <div class={styles.contentGrid}>
        {/* Controls Panel */}
        <div class={styles.controlsPanel}>
          <h3>Parameters</h3>
          <For each={availableProperties()}>
            {(prop) => (
              <div class={styles.controlGroup}>
                <label>{formatPropName(prop.name)}</label>
                <select 
                  class={styles.select}
                  value={selectedValues()[prop.name] || ''}
                  onChange={(e) => handleValueChange(prop.name, e.currentTarget.value)}
                >
                  <option value="">-- Unset --</option>
                  <For each={prop.values}>
                    {(val) => <option value={val}>{val}</option>}
                  </For>
                </select>
              </div>
            )}
          </For>
          
          <button 
            class={styles.resetBtn}
            onClick={() => setSelectedValues({})}
          >
            Reset All Parameters
          </button>
        </div>

        {/* Results Panel */}
        <div class={styles.resultsPanel}>
          <div class={styles.resultsHeader}>
            <h3>Predicted Engagement by Audience Segment</h3>
          </div>

          <div style={{ "display": "flex", "flex-direction": "column", "gap": "32px" }}>
            
            {/* Expert Audience */}
            <div>
              <h4 style={{ "margin-bottom": "16px", "color": "var(--text-secondary)", "display": "flex", "align-items": "center", "gap": "8px" }}>
                <span style={{ "background": "#8b5cf6", "width": "12px", "height": "12px", "border-radius": "50%", "display": "inline-block" }}></span>
                Expert Audience
              </h4>
              <div class={styles.probabilityGrid}>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>High Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Expert.High}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Expert.High}%`, background: "#8b5cf6" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Medium Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Expert.Medium}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Expert.Medium}%`, background: "#a78bfa" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Low Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Expert.Low}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Expert.Low}%`, background: "#c4b5fd" }}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Intermediate Audience */}
            <div>
              <h4 style={{ "margin-bottom": "16px", "color": "var(--text-secondary)", "display": "flex", "align-items": "center", "gap": "8px" }}>
                <span style={{ "background": "#10b981", "width": "12px", "height": "12px", "border-radius": "50%", "display": "inline-block" }}></span>
                Intermediate Audience
              </h4>
              <div class={styles.probabilityGrid}>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>High Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Intermediate.High}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Intermediate.High}%`, background: "#10b981" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Medium Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Intermediate.Medium}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Intermediate.Medium}%`, background: "#34d399" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Low Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Intermediate.Low}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Intermediate.Low}%`, background: "#6ee7b7" }}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Novice Audience */}
            <div>
              <h4 style={{ "margin-bottom": "16px", "color": "var(--text-secondary)", "display": "flex", "align-items": "center", "gap": "8px" }}>
                <span style={{ "background": "#3b82f6", "width": "12px", "height": "12px", "border-radius": "50%", "display": "inline-block" }}></span>
                Novice Audience
              </h4>
              <div class={styles.probabilityGrid}>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>High Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Novice.High}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Novice.High}%`, background: "#3b82f6" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Medium Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Novice.Medium}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Novice.Medium}%`, background: "#60a5fa" }}></div>
                  </div>
                </div>
                <div class={styles.probabilityCard}>
                  <span class={styles.cardTitle}>Low Engagement</span>
                  <span class={styles.probabilityValue}>{calculatedProbabilities().Novice.Low}%</span>
                  <div class={styles.progressBar}>
                    <div class={styles.progressFill} style={{ width: `${calculatedProbabilities().Novice.Low}%`, background: "#93c5fd" }}></div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationMode;
