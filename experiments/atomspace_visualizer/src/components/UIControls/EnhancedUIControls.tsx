// Enhanced UI Controls component with collapsible design
import { Component, createSignal } from 'solid-js';
import styles from './EnhancedUIControls.module.css';

export interface EnhancedUIControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}

const EnhancedUIControls: Component<EnhancedUIControlsProps> = (props) => {
  const [isCollapsed, setIsCollapsed] = createSignal(true);

  return (
    <div class={styles.controlsContainer}>
      <div class={styles.controlsHeader} onClick={() => setIsCollapsed(!isCollapsed())}>
        <h3>Controls</h3>
        <button class={styles.collapseButton}>
          {isCollapsed() ? '▼' : '▲'}
        </button>
      </div>
      
      {!isCollapsed() && (
        <div class={styles.controlsContent}>
          <div class={styles.controlSection}>
            <h4>View Controls</h4>
            <div class={styles.buttonGroup}>
              <button class={styles.controlButton} onClick={props.onZoomIn} title="Zoom In">
                <span>🔍+</span>
                Zoom In
              </button>
              <button class={styles.controlButton} onClick={props.onZoomOut} title="Zoom Out">
                <span>🔍-</span>
                Zoom Out
              </button>
              <button class={styles.controlButton} onClick={props.onReset} title="Reset View">
                <span>↺</span>
                Reset
              </button>
            </div>
          </div>

          <div class={styles.controlSection}>
            <h4>Instructions</h4>
            <ul class={styles.instructionList}>
              <li>🖱️ Click articles or properties to filter</li>
              <li>🖐️ Drag to pan the view</li>
              <li>🔍 Scroll to zoom in/out</li>
              <li>💡 Use legend to explore data</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedUIControls;
