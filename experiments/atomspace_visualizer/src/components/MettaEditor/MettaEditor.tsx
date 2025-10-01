import { Component, For, createSignal, onCleanup } from 'solid-js';

export interface MiningProps {
  conjunctionCount?: number;
  miningApiUrl?: string;
}

type MiningCard = {
  id: string;
  title: string;
  result: unknown;
  x: number;
  y: number;
};

const MettaEditor: Component<MiningProps> = (props) => {
  const [isMining, setIsMining] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [cards, setCards] = createSignal<MiningCard[]>([]);
  let componentActive = true;

  onCleanup(() => {
    componentActive = false;
  });

  const apiBase = () => props.miningApiUrl ?? '/api';

  const toDisplayResult = (result: unknown) => {
    if (typeof result === 'string') return result;
    try {
      return JSON.stringify(result, null, 2);
    } catch {
      return String(result);
    }
  };

  const addCard = (id: string, result: unknown) => {
    setCards((prev) => [
      ...prev,
      {
        id,
        title: `Mining job ${prev.length + 1}`,
        result,
        x: 16 + prev.length * 24,
        y: 16 + prev.length * 24,
      },
    ]);
  };

  const removeCard = (id: string) => {
    setCards((prev) => prev.filter((card) => card.id !== id));
  };

  const updateCardPosition = (id: string, x: number, y: number) => {
    setCards((prev) =>
      prev.map((card) => (card.id === id ? { ...card, x, y } : card))
    );
  };

  const handleDragStart = (event: PointerEvent, id: string) => {
    event.preventDefault();
    const card = cards().find((c) => c.id === id);
    if (!card) return;

    const offsetX = event.clientX - card.x;
    const offsetY = event.clientY - card.y;

    const handleMove = (moveEvent: PointerEvent) => {
      updateCardPosition(id, moveEvent.clientX - offsetX, moveEvent.clientY - offsetY);
    };

    const handleUp = () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  const pollJob = async (jobId: string) => {
    while (componentActive) {
      const response = await fetch(`${apiBase()}/mine/${jobId}`);
      if (!response.ok) throw new Error(`Failed to poll job ${jobId}`);
      const payload = await response.json();
      if (payload.status === 'completed') return payload.result;
      if (payload.status === 'error') throw new Error(payload.error || 'Mining failed');
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new Error('Mining cancelled');
  };

  const startMining = async () => {
    if (isMining()) return;
    setError(null);
    setIsMining(true);

    try {
      const response = await fetch(`${apiBase()}/mine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conjunctionCount: props.conjunctionCount ?? 2 }),
      });
      if (!response.ok) throw new Error('Failed to start mining');
      const payload = await response.json();
      if (!payload?.jobId) throw new Error('Missing job identifier');
      const result = await pollJob(payload.jobId);
      if (!componentActive) return;
      addCard(payload.jobId, result);
    } catch (err) {
      if (!componentActive) return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (componentActive) setIsMining(false);
    }
  };

  return (
    <div style="height:100%; width:100%; position:relative;">
      <style>
        {`
          @keyframes metta-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
          }
        `}
      </style>

      {!isMining() && (
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%);">
          <button
            style="padding:10px 20px; font-size:14px; border:1px solid var(--border-light); border-radius:4px; background:white; cursor:pointer; transition:background 0.2s ease; box-shadow:0 2px 8px rgba(0,0,0,0.1);"
            onClick={startMining}
            onMouseEnter={(event) => (event.currentTarget.style.background = 'var(--bg-primary)')}
            onMouseLeave={(event) => (event.currentTarget.style.background = 'white')}
          >
            Mine
          </button>
        </div>
      )}

      {error() && (
        <div style="position:absolute; top:60%; left:50%; transform:translate(-50%, -50%); padding:8px 16px; border:1px solid #dc2626; border-radius:4px; background:rgba(220,38,38,0.05); color:#dc2626; font-size:12px; margin-top:16px; max-width:80%;">
          {error()}
        </div>
      )}

      {isMining() && (
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center; animation:pulse 2s infinite ease-in-out;">
          <div style="width:36px; height:36px; border:4px solid rgba(203,213,225,0.4); border-top-color:var(--accent, #3b82f6); border-radius:50%; animation:metta-spin 1.2s linear infinite; margin:0 auto 16px;"></div>
          <div style="font-size:16px; font-weight:500; color:#1e293b; margin-bottom:8px;">
            Deep mining in progress...
          </div>
          <div style="font-size:12px; color:#64748b;">
            Extracting precious patterns from the data ore
          </div>
        </div>
      )}

      <For each={cards()}>
        {(card) => (
          <div
            style={`position:absolute; top:${card.y}px; left:${card.x}px; width:260px; max-width:70%; background:white; border:1px solid var(--border-light); border-radius:6px; box-shadow:0 6px 18px rgba(15,23,42,0.15); cursor:grab; z-index:10;`}
            onPointerDown={(event) => handleDragStart(event, card.id)}
          >
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 10px; border-bottom:1px solid var(--border-light); background:var(--bg-primary); border-radius:6px 6px 0 0;">
              <span style="font-size:12px; font-weight:600; color:#1f2937;">{card.title}</span>
              <button
                style="border:none; background:transparent; color:#6b7280; font-size:12px; cursor:pointer;"
                onClick={(event) => {
                  event.stopPropagation();
                  removeCard(card.id);
                }}
              >
                ✕
              </button>
            </div>
            <div style="max-height:220px; overflow:auto; padding:10px; font-size:11px; line-height:1.4; color:#374151;">
              <pre style="margin:0; font-family:'Courier New', Consolas, 'Liberation Mono', Menlo, Courier, monospace; white-space:pre-wrap; word-break:break-word;">
                {toDisplayResult(card.result)}
              </pre>
            </div>
          </div>
        )}
      </For>
    </div>
  );
};

export default MettaEditor;