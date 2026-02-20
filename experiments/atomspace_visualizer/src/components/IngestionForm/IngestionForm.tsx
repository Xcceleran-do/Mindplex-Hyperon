import { Component, createSignal, Show } from 'solid-js';
import styles from './IngestionForm.module.css';
import { API_CONFIG } from '../../config/api';

interface IngestionFormProps {
  onComplete: () => void;
}

const IngestionForm: Component<IngestionFormProps> = (props) => {
  const ingestionEnabled = import.meta.env.VITE_BYPASS_INGESTION === 'false';
  const [username, setUsername] = createSignal('');
  const [isLoading, setIsLoading] = createSignal(false);
  const [error, setError] = createSignal('');
  const [statusMessage, setStatusMessage] = createSignal('');

  const handleIngest = async () => {
    if (!ingestionEnabled) {
      setError('Ingestion is currently bypassed. Set VITE_BYPASS_INGESTION=false to enable it.');
      return;
    }

    if (!username()) {
      setError('Please enter a username');
      return;
    }

    setIsLoading(true);
    setError('');
    setStatusMessage('Connecting to Mindplex...');

    try {
      // Simulate steps for better UX or actually wait for the backend
      setStatusMessage('Fetching articles from Mindplex...');
      
      const response = await fetch(API_CONFIG.getUrl(API_CONFIG.ENDPOINTS.INGEST), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username() }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Ingestion failed');
      }

      const result = await response.json();
      setStatusMessage(result.message || 'Ingestion complete!');
      
      // Short delay to show success message
      setTimeout(() => {
        props.onComplete();
      }, 1000);

    } catch (err: any) {
      setError(err.message);
      setIsLoading(false);
    }
  };

  return (
    <div class={styles.container}>
      <div class={styles.card}>
        <h1 class={styles.title}>Mindplex AtomSpace Visualizer</h1>
        <p class={styles.subtitle}>Enter your Mindplex username to visualize your data</p>
        
        <div class={styles.formGroup}>
          <label for="username" class={styles.label}>Username</label>
          <input
            id="username"
            type="text"
            class={styles.input}
            value={username()}
            onInput={(e) => setUsername(e.currentTarget.value)}
            placeholder="e.g. hruy"
            disabled={isLoading()}
          />
        </div>

        <Show when={error()}>
          <div class={styles.error}>{error()}</div>
        </Show>

        <Show when={statusMessage()}>
          <div class={styles.status}>{statusMessage()}</div>
        </Show>

        <Show when={!ingestionEnabled}>
          <div class={styles.status}>Ingestion is bypassed in this build to preserve the current data.metta.</div>
        </Show>

        <button 
          class={styles.button} 
          onClick={handleIngest}
          disabled={isLoading() || !ingestionEnabled}
        >
          <Show when={isLoading()} fallback="Visualize Data">
            <span class={styles.spinner}></span> Processing...
          </Show>
        </button>
      </div>
    </div>
  );
};

export default IngestionForm;
