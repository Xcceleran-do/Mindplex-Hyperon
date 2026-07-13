import { Component, For, Show, createMemo, createSignal } from 'solid-js';

import {
  simulateEngagement,
  SimulationProofChain,
  SimulationResponse,
  SimulationUnmatchedRule,
} from '../../features/simulation/api';
import { env } from '../../shared/config/env';
import { GraphData } from '../../types';
import styles from './SimulatorPanel.module.css';

interface SimulatorPanelProps {
  graphData: GraphData;
  minedRuleCount: number;
}

interface AttributeRow {
  id: number;
  predicate: string;
  value: string;
  strength: number;
  confidence: number;
}

const DEFAULT_PREDICATES = [
  'tone',
  'length',
  'length-bucket',
  'reading-time',
  'content-type',
  'primary-goal',
  'audience-expertise',
  'audience-sentiment',
  'category',
  'date-period',
  'popularity',
];

const OMITTED_PREDICATES = new Set(['engagement', 'title']);
const ENGAGEMENT_LEVELS = ['High', 'Medium', 'Low'] as const;

const stripQuotes = (value: string) => value.replace(/^"|"$/g, '');

const labelFor = (predicate: string) =>
  predicate
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const formatPercent = (value: number | undefined) => `${Math.round((value ?? 0) * 100)}%`;

const formatScore = (value: number | undefined) =>
  typeof value === 'number' ? value.toFixed(3) : '0.000';

const formatConfidence = (value: number) => `${Math.round(value * 100)}%`;

const truncateProof = (proof: string) => (proof.length > 190 ? `${proof.slice(0, 190)}...` : proof);

const formatRuleTarget = (rule: SimulationUnmatchedRule) => {
  const consequent = rule.consequent;
  return consequent ? `${consequent.predicate} = ${consequent.value}` : rule.rule_id;
};

const chainTitle = (chain: SimulationProofChain, fallbackLevel: string) => {
  const ruleTarget = chain.rule?.consequent?.value || fallbackLevel;
  return `${chain.rule_id || 'proof'} -> ${ruleTarget}`;
};

const SimulatorPanel: Component<SimulatorPanelProps> = (props) => {
  const [rows, setRows] = createSignal<AttributeRow[]>([
    { id: 1, predicate: 'tone', value: '', strength: 1, confidence: 1 },
    { id: 2, predicate: 'reading-time', value: '', strength: 1, confidence: 1 },
  ]);
  const [articleId, setArticleId] = createSignal('');
  const [depth, setDepth] = createSignal(env.defaultChainDepth);
  const [isRunning, setIsRunning] = createSignal(false);
  const [result, setResult] = createSignal<SimulationResponse | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  const valueOptionsByPredicate = createMemo(() => {
    const options = new Map<string, Set<string>>();

    for (const predicate of DEFAULT_PREDICATES) {
      if (!OMITTED_PREDICATES.has(predicate)) {
        options.set(predicate, new Set<string>());
      }
    }

    for (const node of props.graphData.nodes) {
      const predicate = node.metadata.propertyName;
      if (!predicate || OMITTED_PREDICATES.has(predicate)) {
        continue;
      }
      const rawValue = node.metadata.originalExpression || node.label;
      if (!rawValue) {
        continue;
      }
      if (!options.has(predicate)) {
        options.set(predicate, new Set<string>());
      }
      options.get(predicate)!.add(stripQuotes(rawValue));
    }

    return options;
  });

  const predicateOptions = createMemo(() =>
    Array.from(valueOptionsByPredicate().keys()).sort((left, right) => {
      const leftIndex = DEFAULT_PREDICATES.indexOf(left);
      const rightIndex = DEFAULT_PREDICATES.indexOf(right);
      if (leftIndex >= 0 && rightIndex >= 0) return leftIndex - rightIndex;
      if (leftIndex >= 0) return -1;
      if (rightIndex >= 0) return 1;
      return left.localeCompare(right);
    })
  );

  const valuesFor = (predicate: string) =>
    Array.from(valueOptionsByPredicate().get(predicate) || []).sort((left, right) =>
      left.localeCompare(right)
    );

  const updateRow = (rowId: number, patch: Partial<AttributeRow>) => {
    setRows((current) =>
      current.map((row) => {
        if (row.id !== rowId) return row;
        const next = { ...row, ...patch };
        if (patch.predicate) {
          next.value = valuesFor(patch.predicate)[0] || '';
        }
        return next;
      })
    );
  };

  const addRow = () => {
    const predicate = predicateOptions()[0] || 'tone';
    setRows((current) => [
      ...current,
      {
        id: Date.now(),
        predicate,
        value: valuesFor(predicate)[0] || '',
        strength: 1,
        confidence: 1,
      },
    ]);
  };

  const removeRow = (rowId: number) => {
    setRows((current) => current.filter((row) => row.id !== rowId));
  };

  const selectedAttributes = createMemo(() => {
    const attributes: Record<string, { value: string; strength: number; confidence: number }> = {};
    for (const row of rows()) {
      if (!row.predicate || !row.value) {
        continue;
      }
      attributes[row.predicate] = {
        value: row.value,
        strength: row.strength,
        confidence: row.confidence,
      };
    }
    return attributes;
  });

  const hasMinedRules = createMemo(() => props.minedRuleCount > 0);
  const canRun = createMemo(() =>
    hasMinedRules() && Object.keys(selectedAttributes()).length > 0 && !isRunning()
  );

  const runSimulation = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const response = await simulateEngagement({
        article_id: articleId() || undefined,
        attributes: selectedAttributes(),
        depth: depth(),
      });
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Simulation failed.';
      setError(message);
    } finally {
      setIsRunning(false);
    }
  };

  const winningBucket = createMemo(() => {
    const current = result();
    if (!current) return null;
    return current.buckets[current.predicted_engagement as 'High' | 'Medium' | 'Low'] || null;
  });

  const winningChains = createMemo(() => {
    const current = result();
    if (!current?.explanation) return [];
    return current.explanation.chains_by_level[current.predicted_engagement as 'High' | 'Medium' | 'Low'] || [];
  });

  const usefulUnmatchedRules = createMemo(() => {
    const current = result();
    if (!current?.explanation) return [];
    return current.explanation.unmatched_rules
      .filter((rule) => rule.missing_antecedents.length > 0)
      .slice(0, 4);
  });

  const conditionalSuggestions = createMemo(() => {
    const current = result();
    if (!current?.explanation) return [];
    const levelSuggestions =
      current.explanation.conditional_suggestions_by_level?.[
        current.predicted_engagement as 'High' | 'Medium' | 'Low'
      ] || [];
    return (levelSuggestions.length > 0 ? levelSuggestions : current.explanation.conditional_suggestions || [])
      .filter((suggestion) => suggestion.missing_antecedents.length > 0)
      .slice(0, 4);
  });

  return (
    <div class={styles.simulatorPanel}>
      <div class={styles.topLine}>
        <div>
          <div class={styles.kicker}>What-if engine</div>
          <div class={styles.title}>Engagement Simulator</div>
        </div>
        <div class={styles.ruleBadge}>{props.minedRuleCount} rules</div>
      </div>

      <Show when={!hasMinedRules()}>
        <div class={styles.guidanceBox}>Mine rules first. The simulator needs PeTTa rules before it can score a hypothetical article.</div>
      </Show>

      <div class={styles.controlsGrid}>
        <label class={styles.field}>
          <span>Article id</span>
          <input
            value={articleId()}
            onInput={(event) => setArticleId(event.currentTarget.value)}
            placeholder="Auto-generated"
          />
        </label>
        <label class={styles.field}>
          <span>Depth</span>
          <input
            type="number"
            min="1"
            value={depth()}
            onInput={(event) => setDepth(Math.max(1, Number(event.currentTarget.value) || 1))}
          />
        </label>
      </div>

      <div class={styles.attributeList}>
        <For each={rows()}>
          {(row) => (
            <div class={styles.attributeCard}>
              <div class={styles.attributeRow}>
                <select
                  value={row.predicate}
                  onChange={(event) => updateRow(row.id, { predicate: event.currentTarget.value })}
                  aria-label="Attribute"
                >
                  <For each={predicateOptions()}>
                    {(predicate) => <option value={predicate}>{labelFor(predicate)}</option>}
                  </For>
                </select>

                <Show
                  when={valuesFor(row.predicate).length > 0}
                  fallback={
                    <input
                      value={row.value}
                      onInput={(event) => updateRow(row.id, { value: event.currentTarget.value })}
                      placeholder="Value"
                      aria-label="Attribute value"
                    />
                  }
                >
                  <select
                    value={row.value}
                    onChange={(event) => updateRow(row.id, { value: event.currentTarget.value })}
                    aria-label="Attribute value"
                  >
                    <option value="">Choose value</option>
                    <For each={valuesFor(row.predicate)}>
                      {(value) => <option value={value}>{value}</option>}
                    </For>
                  </select>
                </Show>

                <button
                  type="button"
                  class={styles.removeButton}
                  onClick={() => removeRow(row.id)}
                  aria-label="Remove attribute"
                  title="Remove attribute"
                >
                  <span aria-hidden="true">x</span>
                </button>
              </div>
              <label class={styles.confidenceControl}>
                <span>Fact confidence {formatConfidence(row.confidence)}</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={row.confidence}
                  onInput={(event) => updateRow(row.id, { confidence: Number(event.currentTarget.value) })}
                />
              </label>
            </div>
          )}
        </For>
      </div>

      <div class={styles.actionRow}>
        <button type="button" class={styles.addButton} onClick={addRow}>
          Add Attribute
        </button>
        <button type="button" class={styles.runButton} onClick={runSimulation} disabled={!canRun()}>
          <Show when={isRunning()} fallback="Run Simulation">
            Running...
          </Show>
        </button>
      </div>

      <Show when={error()}>
        {(message) => <div class={styles.errorBox}>{message()}</div>}
      </Show>

      <Show when={result()}>
        {(simulation) => (
          <div class={styles.resultPanel}>
            <div class={styles.predictionRow}>
              <span>Prediction</span>
              <strong>{simulation().predicted_engagement}</strong>
            </div>

            <div class={styles.probabilityList}>
              <For each={ENGAGEMENT_LEVELS}>
                {(level) => (
                  <div class={styles.probabilityItem}>
                    <div class={styles.probabilityMeta}>
                      <span>{level}</span>
                      <span>{formatPercent(simulation().probabilities[level])}</span>
                    </div>
                    <div class={styles.probabilityTrack}>
                      <div
                        class={`${styles.probabilityFill} ${styles[`fill${level}`]}`}
                        style={{ width: formatPercent(simulation().probabilities[level]) }}
                      />
                    </div>
                    <div class={styles.proofMeta}>
                      {simulation().buckets[level].proof_count} proofs, score{' '}
                      {formatScore(simulation().buckets[level].raw_score)}
                      <Show when={(simulation().buckets[level].conditional_score || 0) > 0}>
                        {' '}· conditional {formatScore(simulation().buckets[level].conditional_score)}
                      </Show>
                    </div>
                  </div>
                )}
              </For>
            </div>

            <div class={styles.resultFooter}>
              <span>{simulation().rules_used} rules used</span>
              <span>{simulation().used_prior_fallback ? 'Prior fallback' : 'Rule proofs'}</span>
            </div>

            <Show when={simulation().explanation?.summary}>
              {(summary) => (
                <div class={simulation().used_prior_fallback ? styles.fallbackBox : styles.explanationBox}>
                  {summary()}
                </div>
              )}
            </Show>

            <Show when={winningBucket()?.aggregated_stv}>
              {(stv) => (
                <div class={styles.stvBox}>
                  STV strength {formatScore(stv().strength)} · confidence {formatScore(stv().confidence)}
                </div>
              )}
            </Show>

            <Show when={!winningBucket()?.aggregated_stv && winningBucket()?.conditional_stv}>
              {(stv) => (
                <div class={styles.stvBox}>
                  Conditional STV strength {formatScore(stv().strength)} · confidence {formatScore(stv().confidence)}
                </div>
              )}
            </Show>

            <Show when={(winningBucket()?.proofs.length || 0) > 0}>
              <details class={styles.proofDetails}>
                <summary>Winning proof sample</summary>
                <pre>{truncateProof(winningBucket()!.proofs[0])}</pre>
              </details>
            </Show>

            <Show when={winningChains().length > 0}>
              <div class={styles.chainList}>
                <div class={styles.chainHeading}>Inference chains</div>
                <For each={winningChains()}>
                  {(chain) => (
                    <details class={styles.chainItem} open>
                      <summary>{chainTitle(chain, simulation().predicted_engagement)}</summary>
                      <div class={styles.chainBody}>
                        <Show when={chain.rule}>
                          {(rule) => (
                            <div class={styles.chainBlock}>
                              <span>Rule</span>
                              <pre>{rule().atom}</pre>
                            </div>
                          )}
                        </Show>
                        <Show when={chain.facts.length > 0}>
                          <div class={styles.chainBlock}>
                            <span>Facts used</span>
                            <For each={chain.facts}>
                              {(fact) => <code>{fact.atom}</code>}
                            </For>
                          </div>
                        </Show>
                        <Show when={chain.stv}>
                          {(stv) => (
                            <div class={styles.chainStv}>
                              Final STV {formatScore(stv().strength)} / {formatScore(stv().confidence)}
                            </div>
                          )}
                        </Show>
                        <div class={styles.chainBlock}>
                          <span>Raw proof</span>
                          <pre>{truncateProof(chain.proof)}</pre>
                        </div>
                      </div>
                    </details>
                  )}
                </For>
              </div>
            </Show>

            <Show when={simulation().used_prior_fallback && usefulUnmatchedRules().length > 0}>
              <div class={styles.chainList}>
                <div class={styles.chainHeading}>Why no rule fired</div>
                <For each={usefulUnmatchedRules()}>
                  {(rule) => (
                    <div class={styles.missingRule}>
                      <div class={styles.missingRuleTitle}>{rule.rule_id}: {formatRuleTarget(rule)}</div>
                      <Show when={rule.missing_antecedents.length > 0}>
                        <div class={styles.chainBlock}>
                          <span>Missing facts</span>
                          <For each={rule.missing_antecedents}>
                            {(missing) => <code>{missing}</code>}
                          </For>
                        </div>
                      </Show>
                      <Show when={rule.matched_antecedents.length > 0}>
                        <div class={styles.chainBlock}>
                          <span>Matched facts</span>
                          <For each={rule.matched_antecedents}>
                            {(matched) => <code>{matched.fact.atom}</code>}
                          </For>
                        </div>
                      </Show>
                    </div>
                  )}
                </For>
              </div>
            </Show>

            <Show when={conditionalSuggestions().length > 0}>
              <div class={styles.chainList}>
                <div class={styles.chainHeading}>Suggested missing attributes</div>
                <For each={conditionalSuggestions()}>
                  {(suggestion) => (
                    <div class={styles.suggestionRule}>
                      <div class={styles.missingRuleTitle}>
                        {suggestion.rule_id}: {formatRuleTarget(suggestion)}
                      </div>
                      <div class={styles.suggestionMeta}>
                        {suggestion.summary} STV {formatScore(suggestion.conditional_stv.strength)} /{' '}
                        {formatScore(suggestion.conditional_stv.confidence)}
                      </div>
                      <div class={styles.chainBlock}>
                        <span>Add these facts</span>
                        <For each={suggestion.missing_antecedents}>
                          {(missing) => <code>{missing}</code>}
                        </For>
                      </div>
                      <Show when={suggestion.matched_antecedents.length > 0}>
                        <div class={styles.chainBlock}>
                          <span>Already matched</span>
                          <For each={suggestion.matched_antecedents}>
                            {(matched) => <code>{matched.fact.atom}</code>}
                          </For>
                        </div>
                      </Show>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </div>
        )}
      </Show>
    </div>
  );
};

export default SimulatorPanel;
