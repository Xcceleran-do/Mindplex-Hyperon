// Context Menu component for node interactions
import { Component, createSignal, Show } from 'solid-js';
import { GraphNode } from '../../types';

export interface ContextMenuProps {
  node: GraphNode | null;
  position: { x: number; y: number } | null;
  onIsolate: (node: GraphNode) => void;
  onCopyLabel: (node: GraphNode) => void;
  onClose: () => void;
}

const ContextMenu: Component<ContextMenuProps> = (props) => {
  return (
    <Show when={props.node && props.position}>
      <div
        class="context-menu"
        style={`
          position: fixed;
          left: ${props.position?.x}px;
          top: ${props.position?.y}px;
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 6px;
          box-shadow: var(--card-shadow);
          backdrop-filter: blur(10px);
          z-index: 2000;
          padding: 4px 0;
          min-width: 120px;
        `}
      >
        <div
          class="context-menu-item"
          style="padding: 8px 16px; cursor: pointer; font-size: 14px; color: var(--text-primary); transition: background-color 0.2s ease;"
          onClick={() => {
            if (props.node) props.onIsolate(props.node);
            props.onClose();
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Isolate
        </div>
        <div
          class="context-menu-item"
          style="padding: 8px 16px; cursor: pointer; font-size: 14px; color: var(--text-primary); transition: background-color 0.2s ease;"
          onClick={() => {
            if (props.node) props.onCopyLabel(props.node);
            props.onClose();
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-primary)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Copy Label
        </div>
      </div>
    </Show>
  );
};

export default ContextMenu;