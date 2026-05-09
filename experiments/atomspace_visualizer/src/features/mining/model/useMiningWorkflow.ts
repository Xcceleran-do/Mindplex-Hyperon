import { createSignal } from 'solid-js';
import type { FilterState, GraphData } from '../../../types';
import { minePatterns, type PatternResult } from '../api';

export type MiningNotice = {
  type: 'info' | 'success' | 'error';
  text: string;
};

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

  const startMining = async (conjunctSize: number, minSupport = 3) => {
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

      const patterns = Array.isArray(job.result) ? job.result : [];
      setMiningResults(patterns);
      setMiningNotice({
        type: 'success',
        text: `Mining completed with ${patterns.length} pattern${patterns.length === 1 ? '' : 's'}.`,
      });
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
    setMiningResults(patterns);
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
