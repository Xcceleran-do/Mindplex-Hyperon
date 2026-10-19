import { Component, createSignal, Show } from 'solid-js';
import styles from './IngestionForm.module.css';
import { ingestUserData } from '../../features/ingestion/api';
import { env } from '../../shared/config/env';

interface IngestionFormProps {
  onComplete: () => void;
}

const IngestionForm: Component<IngestionFormProps> = (props) => {
  const ingestionEnabled = env.ingestionEnabled;
  const [username, setUsername] = createSignal('');
  const [isLoading, setIsLoading] = createSignal(false);
  const [error, setError] = createSignal('');
  const [statusMessage, setStatusMessage] = createSignal('');

  const handleIngest = async () => {
    if (!ingestionEnabled) {
      setError('Ingestion is disabled. Set INGESTION_ENABLED=true and rebuild the frontend to enable it.');
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
      
      const result = await ingestUserData(username());
      setStatusMessage(result.message || 'Ingestion complete!');
      props.onComplete();

    } catch (err: any) {
      setError(err.message);
      setIsLoading(false);
    }
  };

  return (
    <div class={styles.container}>
      <div class={styles.backgroundGrid} aria-hidden="true" />
      <div class={styles.card}>
        <div class={styles.brandMark}>M</div>
        <div class={styles.kicker}>Mindplex Hyperon</div>
        <h1 class={styles.title}>Turn articles into a reasoning workspace.</h1>
        <p class={styles.subtitle}>
          Load a Mindplex author, mine PeTTa rules, simulate hypothetical articles, and ask the backward chainer for proof.
        </p>

        <div class={styles.featureGrid} aria-label="Workflow">
          <div><strong>1</strong><span>Ingest posts</span></div>
          <div><strong>2</strong><span>Mine rules</span></div>
          <div><strong>3</strong><span>Simulate outcomes</span></div>
        </div>

        <div class={styles.formGroup}>
          <label for="username" class={styles.label}>Mindplex username</label>
          <div class={styles.inputShell}>
            <span>@</span>
            <input
              id="username"
              type="text"
              class={styles.input}
              value={username()}
              onInput={(e) => setUsername(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleIngest();
                }
              }}
              placeholder="author_handle"
              disabled={isLoading()}
            />
          </div>
        </div>

        <Show when={error()}>
          <div class={styles.error}>{error()}</div>
        </Show>

        <Show when={statusMessage()}>
          <div class={styles.status}>{statusMessage()}</div>
        </Show>

        <Show when={!ingestionEnabled}>
          <div class={styles.status}>Ingestion is disabled in this build; the existing data.metta will be used.</div>
        </Show>

        <button
          class={styles.button}
          onClick={handleIngest}
          disabled={isLoading() || !ingestionEnabled}
        >
          <Show when={isLoading()} fallback="Open Workspace">
            <span class={styles.spinner}></span> Processing...
          </Show>
        </button>
        <p class={styles.footerNote}>The demo builds a fresh `data.metta` file, then opens the visual reasoning canvas.</p>
      </div>
    </div>
  );
};

export default IngestionForm;
