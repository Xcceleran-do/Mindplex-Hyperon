import { createSignal, createEffect, Show, onMount } from 'solid-js';
import './MiningInterface.css';

export interface MiningResult {
  jobId: string;
  status: 'running' | 'completed' | 'error';
  result?: any;
  error?: string;
  duration?: number;
}

export interface MiningInterfaceProps {
  onMiningStart?: () => void;
  onMiningComplete?: (result: MiningResult) => void;
}

const MiningInterface = (props: MiningInterfaceProps) => {
  const [isMining, setIsMining] = createSignal(false);
  const [miningResult, setMiningResult] = createSignal<MiningResult | null>(null);
  const [conjunctionCount, setConjunctionCount] = createSignal(3);
  const [showResult, setShowResult] = createSignal(false);
  
  // Drag functionality
  const [isDragging, setIsDragging] = createSignal(false);
  const [dragOffset, setDragOffset] = createSignal({ x: 0, y: 0 });
  const [position, setPosition] = createSignal({ x: 0, y: 0 });
  let miningInterfaceRef: HTMLDivElement | undefined;

  const startMining = async () => {
    setIsMining(true);
    setMiningResult(null);
    setShowResult(false);
    props.onMiningStart?.();

    try {
      // Start mining job
      const response = await fetch('http://localhost:5000/api/mine', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ conjunctionCount: conjunctionCount() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const jobData = await response.json();
      const jobId = jobData.jobId;

      // Poll for results
      const pollForResults = async () => {
        try {
          const statusResponse = await fetch(`http://localhost:5000/api/mine/${jobId}`);
          if (!statusResponse.ok) {
            throw new Error(`HTTP error! status: ${statusResponse.status}`);
          }

          const statusData = await statusResponse.json();
          
          if (statusData.status === 'completed' || statusData.status === 'error') {
            setIsMining(false);
            setMiningResult(statusData);
            setShowResult(true);
            props.onMiningComplete?.(statusData);
          } else {
            // Continue polling
            setTimeout(pollForResults, 1000);
          }
        } catch (error) {
          console.error('Error polling for results:', error);
          setIsMining(false);
          setMiningResult({
            jobId: jobId,
            status: 'error',
            error: `Polling error: ${error instanceof Error ? error.message : 'Unknown error'}`
          });
          setShowResult(true);
        }
      };

      // Start polling
      setTimeout(pollForResults, 1000);

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

  // Drag functionality
  const handleMouseDown = (e: MouseEvent) => {
    if (!miningInterfaceRef) return;
    
    setIsDragging(true);
    const rect = miningInterfaceRef.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    e.preventDefault();
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging()) return;

    const newX = e.clientX - dragOffset().x;
    const newY = e.clientY - dragOffset().y;

    // Constrain to viewport bounds
    const maxX = window.innerWidth - (miningInterfaceRef?.offsetWidth || 0);
    const maxY = window.innerHeight - (miningInterfaceRef?.offsetHeight || 0);

    setPosition({
      x: Math.max(0, Math.min(newX, maxX)),
      y: Math.max(0, Math.min(newY, maxY))
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  onMount(() => {
    // Clean up event listeners on unmount
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  });

  return (
    <div 
      ref={miningInterfaceRef}
      class={`mining-interface ${isDragging() ? 'dragging' : ''}`}
      style={position().x !== 0 || position().y !== 0 ? 
        `transform: translate(${position().x}px, ${position().y}px) translateX(-50%);` : 
        ''
      }
      onMouseDown={handleMouseDown}
    >
      {/* Mining Control Panel */}
      <div class="mining-controls">
        <div class="conjunction-input">
          <span class="drag-indicator">⋮⋮</span>
          <label for="conjunction-count">Conjunction Count:</label>
          <input
            id="conjunction-count"
            type="number"
            min="1"
            max="10"
            value={conjunctionCount()}
            onInput={(e) => setConjunctionCount(parseInt(e.target.value) || 3)}
            disabled={isMining()}
          />
        </div>
        
        <button
          class={`mine-button ${isMining() ? 'mining' : ''}`}
          onClick={startMining}
          disabled={isMining()}
        >
          <div class="button-content">
            <Show when={!isMining()}>
              <span class="pickaxe-icon">⛏️</span>
              <span class="button-text">Mine the Gold</span>
              <span class="gems-icon">💎</span>
            </Show>
            <Show when={isMining()}>
              <div class="mining-animation">
                <div class="pickaxe-swing">⛏️</div>
                <span class="mining-text">Mining...</span>
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

      {/* Mining Animation Overlay */}
      <Show when={isMining()}>
        <div class="mining-overlay">
          <div class="mining-scene">
            <div class="cave-entrance">🕳️</div>
            <div class="miner">
              <div class="miner-body">👷</div>
              <div class="pickaxe-animation">⛏️</div>
            </div>
            <div class="ore-particles">
              <div class="particle">⚡</div>
              <div class="particle">💎</div>
              <div class="particle">🔥</div>
              <div class="particle">✨</div>
            </div>
            <div class="progress-bar">
              <div class="progress-fill"></div>
            </div>
            <div class="mining-status">
              <p>Deep mining in progress...</p>
              <p class="sub-text">Extracting precious patterns from the data ore</p>
            </div>
          </div>
        </div>
      </Show>

      {/* Result Card */}
      <Show when={showResult() && miningResult()}>
        <div class="result-overlay" onClick={closeResult}>
          <div class="result-card" onClick={(e) => e.stopPropagation()}>
            <Show when={miningResult()?.status === 'completed'}>
              <div class="result-header gold">
                <h2>🏆 The Gold 🏆</h2>
                <p class="subtitle">Precious patterns extracted from the depths</p>
              </div>
              <div class="result-content">
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