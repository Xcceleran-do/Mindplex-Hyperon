import { createSignal } from 'solid-js';
import type { FilterState, GraphData } from '../../../types';
import { minePatterns, type PatternResult } from '../api';
import { DEFAULT_MIN_SUPPORT } from '../defaults';

export type MiningNotice = {
  type: 'info' | 'error';
  text: string;
};

const supportValue = (pattern: PatternResult) => {
  const value = Number.parseFloat(pattern.support);
  return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
};

export const sortPatternsBySupport = (patterns: PatternResult[]) => [...patterns].sort(
  (left, right) => supportValue(right) - supportValue(left) || left.pattern.localeCompare(right.pattern),
);

export const useMiningWorkflow = (
  graphData: () => GraphData,
  onFilterChange: (filter: FilterState) => void,
) => {
  const [miningResults, setMiningResults] = createSignal<PatternResult[]>([]);
  const [currentConjunctSize, setCurrentConjunctSize] = createSignal<number | undefined>(undefined);
  const [miningNotice, setMiningNotice] = createSignal<MiningNotice | null>(null);

  let animationInterval: number | undefined;

  const startMiningAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
    }

    const articles = graphData()
      .nodes
      .filter((node) => node.metadata.columnType === 'article')
      .map((node) => node.metadata.originalExpression || node.label);

    if (articles.length === 0) {
      return;
    }

    let currentIndex = 0;
    animationInterval = setInterval(() => {
      const currentArticle = articles[currentIndex % articles.length];
      onFilterChange({
        active: true,
        articleIds: [currentArticle],
        propertyFilters: [],
      });
      currentIndex += 1;
    }, 1000) as unknown as number;
  };

  const stopMiningAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = undefined;
    }

    onFilterChange({
      active: false,
      articleIds: [],
      propertyFilters: [],
    });
  };

  const startMining = async (conjunctSize: number, minSupport = DEFAULT_MIN_SUPPORT) => {
    try {
      startMiningAnimation();
      setMiningNotice(null);

      const job = await minePatterns(conjunctSize, minSupport);
      stopMiningAnimation();
      setCurrentConjunctSize(conjunctSize);

      if (job.status === 'no_results') {
        setMiningResults([]);
        setMiningNotice({
          type: 'info',
          text: job.message || 'No patterns found. Try lowering MinSup or conjunction count.',
        });
        return;
      }

      const patterns = sortPatternsBySupport(Array.isArray(job.result) ? job.result : []);
      setMiningResults(patterns);
      setMiningNotice(null);
    } catch (error) {
      console.error('Mining error:', error);
      stopMiningAnimation();
      setMiningResults([]);
      setMiningNotice({
        type: 'error',
        text: error instanceof Error ? error.message : 'Mining failed. Please try again.',
      });
    }
  };

  const setPatternsFound = (patterns: PatternResult[], conjunctSize?: number) => {
    stopMiningAnimation();
    setMiningResults(sortPatternsBySupport(patterns));
    if (conjunctSize) {
      setCurrentConjunctSize(conjunctSize);
    }
  };

  return {
    miningResults,
    currentConjunctSize,
    miningNotice,
    startMining,
    setPatternsFound,
    stopMiningAnimation,
  };
};
