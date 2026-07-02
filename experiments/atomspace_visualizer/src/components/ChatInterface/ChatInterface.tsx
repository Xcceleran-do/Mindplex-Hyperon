import { createSignal, For, Show, createEffect } from 'solid-js';
import { analyzePattern, sendChatMessage, summarizePatterns, type ChatHistoryMessage } from '../../features/chat/api';
import './ChatInterface.css';

const AssistantGlyph = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 3 4.8 7.1v8.2L12 19.5l7.2-4.2V7.1L12 3Z" />
    <path d="M8.2 10.2h7.6M8.2 13.8h7.6M12 7.4v9.2" />
    <circle cx="12" cy="12" r="1.6" />
  </svg>
);

const ClearIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M5 7h14M9 7V5.5h6V7M9 10v7M15 10v7M7 7l1 13h8l1-13" />
  </svg>
);

const SendIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="m4 12 15-7-4.8 14-2.9-5.4L4 12Z" />
    <path d="m11.3 13.6 3.1-3.8" />
  </svg>
);

const summarizeFunctionCalls = (functionCalls: unknown): string => {
  if (!Array.isArray(functionCalls) || functionCalls.length === 0) {
    return '';
  }

  const parts = functionCalls
    .map((item) => {
      const call = item as { name?: unknown; result?: any };
      if (typeof call.name !== 'string') {
        return null;
      }
      if (call.name === 'getChainerResult' && call.result && typeof call.result === 'object') {
        const status = call.result.status ? String(call.result.status) : 'unknown';
        const proofs = typeof call.result.proof_count === 'number' ? `, ${call.result.proof_count} proofs` : '';
        return `${call.name} (${status}${proofs})`;
      }
      if (call.result && typeof call.result === 'object' && typeof call.result.status === 'string') {
        return `${call.name} (${call.result.status})`;
      }
      return call.name;
    })
    .filter((part): part is string => Boolean(part));

  return parts.length > 0 ? `Functions: ${parts.join(' -> ')}` : '';
};

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  conjunct?: {
    pattern: string;
    support: string;
    summary?: string;
  };
  functionCalls?: unknown;
  isTyping?: boolean;
}

export interface ChatInterfaceProps {
  onVisualize: (filter: import('../../types').FilterState) => void;
  miningResults?: Array<{ pattern: string; support: string }>;
  conjunctSize?: number;
  onMiningStart?: (conjunctSize: number, minSupport?: number) => void | Promise<void>;
  onPatternsFound?: (patterns: Array<{ pattern: string; support: string }>, conjunctSize?: number) => void;
  onShowRules?: () => void;
  isOpen?: boolean;
  onClose?: () => void;
}

const ChatInterface = (props: ChatInterfaceProps) => {
  const [messages, setMessages] = createSignal<Message[]>([]);
  const [inputText, setInputText] = createSignal('');
  const [isLoading, setIsLoading] = createSignal(false);
  const [isMinimized, setIsMinimized] = createSignal(false);
  const [lastConjunctSize, setLastConjunctSize] = createSignal<number | null>(null);

  let chatContainerRef: HTMLDivElement | undefined;
  let inputRef: HTMLTextAreaElement | undefined;

  console.log('ChatInterface rendered, props:', { conjunctSize: props.conjunctSize });

  // NOTE: We intentionally do NOT auto-trigger mining when `props.conjunctSize`
  // changes because that value is also set by the parent when a mining job
  // completes. Doing so could cause a feedback loop where the parent starts
  // mining and the chat re-requests mining again. Instead, user-initiated
  // chat commands are handled inside sendMessage() where we can differentiate
  // direct user intent from parent-driven state updates.

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    if (chatContainerRef) {
      chatContainerRef.scrollTop = chatContainerRef.scrollHeight;
    }
  };

  createEffect(() => {
    messages(); // Track messages changes
    setTimeout(scrollToBottom, 100);
  });

  // Process mining results and add them to chat
  createEffect(() => {
    const results = props.miningResults;
    if (results && results.length > 0) {
      // Auto-open chat UI when mining completes if parent provided control
      if (props.isOpen === undefined || props.isOpen === false) {
        // If onClose/onOpen handlers exist on parent, prefer that
        // We open by calling onMiningStart? Instead expose onOpen via adding message
      }
      // Add system message about mining completion
      const systemMsg: Message = {
        id: `sys-${Date.now()}`,
        role: 'system',
        content: `Mining completed! Found ${results.length} pattern(s). Analyzing results...`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, systemMsg]);

      // Request a single summary for all patterns and display it
      (async () => {
        try {
          const data = await summarizePatterns(results);
          const assistantMsg: Message = {
            id: `msg-${Date.now()}-${Math.random()}`,
            role: 'assistant',
            content: data.summary,
            timestamp: new Date()
          };
          setMessages(prev => [...prev, assistantMsg]);
        } catch (e) {
          console.error('Error fetching summary:', e);
          // Fallback behavior
          results.forEach((result, index) => {
            setTimeout(() => {
              analyzeConjunct(result.pattern, result.support);
            }, index * 500);
          });
        }
      })();
    }
  });

  const analyzeConjunct = async (pattern: string, support: string) => {
    try {
      const data = await analyzePattern(pattern, support);

      const assistantMsg: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: data.summary,
        timestamp: new Date(),
        conjunct: {
          pattern,
          support,
          summary: data.summary
        }
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error('Error analyzing conjunct:', error);

      // Fallback to simple summary
      const fallbackMsg: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: `**Pattern Found** (Support: ${support})\n\n\`\`\`\n${pattern}\n\`\`\`\n\nThis pattern appears ${support} times in the dataset.`,
        timestamp: new Date(),
        conjunct: { pattern, support }
      };

      setMessages(prev => [...prev, fallbackMsg]);
    }
  };

  const applyFunctionCallEffects = (functionCalls: unknown) => {
    if (!Array.isArray(functionCalls)) {
      return;
    }

    const miningNames = new Set(['mine_pattern', 'start_mining_job', 'startMiningJob', 'minePattern']);

    for (const functionCall of functionCalls) {
      const call = functionCall as {
        name?: unknown;
        args?: Record<string, unknown>;
        result?: any;
      };
      if (typeof call.name !== 'string' || !miningNames.has(call.name)) {
        continue;
      }

      const result = call.result;
      const minedPayload = result?.result && typeof result.result === 'object' ? result.result : result;
      const candidatePatterns = Array.isArray(minedPayload?.patterns)
        ? minedPayload.patterns
        : Array.isArray(result?.result)
          ? result.result
          : [];

      const patterns = candidatePatterns
        .filter((item: any) => typeof item?.pattern === 'string')
        .map((item: any) => ({
          pattern: item.pattern,
          support: String(item.support ?? ''),
        }));

      if (patterns.length === 0) {
        continue;
      }

      const rawConjunctSize =
        minedPayload?.conjunction_count
        ?? result?.conjunction_count
        ?? call.args?.conjunction_count
        ?? call.args?.numberOfConjunction;
      const conjunctSize = typeof rawConjunctSize === 'number'
        ? rawConjunctSize
        : typeof rawConjunctSize === 'string'
          ? parseInt(rawConjunctSize, 10)
          : undefined;

      props.onPatternsFound?.(patterns, Number.isFinite(conjunctSize) ? conjunctSize : undefined);
      props.onShowRules?.();
    }
  };

  // Send AI message (internal function)
  const sendAIMessage = async (text: string) => {
    // Show typing indicator
    const typingMsg: Message = {
      id: `typing-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isTyping: true
    };
    setMessages(prev => [...prev, typingMsg]);
    setIsLoading(true);

    try {
      const history: ChatHistoryMessage[] = messages()
        .filter(m => !m.isTyping && m.role !== 'system')
        .map(m => ({
          role: m.role,
          content: m.functionCalls
            ? `${m.content}\n\n[${summarizeFunctionCalls(m.functionCalls)}]`
            : m.content
        }));
      const data = await sendChatMessage(text, history);
      applyFunctionCallEffects(data.functionCalls);

      // Remove typing indicator and add actual response
      setMessages(prev => prev.filter(m => !m.isTyping));

      const aiMsg: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        functionCalls: data.functionCalls
      };

      setMessages(prev => [...prev, aiMsg]);

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => prev.filter(m => !m.isTyping));
      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Send message to AI (from user input)
  const sendMessage = async () => {
    const text = inputText().trim();
    if (!text) return;

    // Add user message
    const userMsg: Message = {
      id: `msg-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMsg]);
    setInputText('');

    // Get AI response
    await sendAIMessage(text);
  };

  const handleVisualize = (pattern: string) => {
    // If pattern is a string, parse it to FilterState
    if (typeof pattern === 'string') {
      const regex = /(\w+) \$\w+ "([^"]+)"/g;
      const propertyFilters = [];
      let match;
      while ((match = regex.exec(pattern)) !== null) {
        propertyFilters.push({ property: `${match[1]}`, value: `"${match[2]}"` });
      }
      props.onVisualize({
        active: true,
        propertyFilters,
        articleIds: [],
      });
    } else {
      // If already a FilterState, just pass through
      props.onVisualize(pattern);
    }
  };

  const handlePatternClick = (e: MouseEvent, message: Message) => {
    const target = e.target as HTMLElement;

    // Check if clicked element is a pattern reference
    if (target.classList.contains('pattern-ref')) {
      const patternIndex = parseInt(target.getAttribute('data-pattern') || '0');

      // Get the pattern from mining results
      if (props.miningResults && props.miningResults.length >= patternIndex) {
        const patternObj = props.miningResults[patternIndex - 1];
        if (patternObj) {
          // Parse pattern string to extract property-value pairs
          // Example pattern: (length-bucket $x "low") (tone $x "Analytical")
          // Updated regex to handle variables with special chars and properties with hyphens
          const regex = /\(([^\s()]+)\s+\$[^\s()]+\s+("|'|)([^"')]+)\2\)/g;
          const propertyFilters: { property: string; value: string }[] = [];
          let match;
          while ((match = regex.exec(patternObj.pattern)) !== null) {
            const property = match[1];
            const value = `"${match[3]?.trim()}"`;
            if (property && value) {
              propertyFilters.push({ property, value });
            }
          }
          console.log('Pattern:', patternObj.pattern);
          console.log('Extracted propertyFilters:', propertyFilters);
          // Set filter state to visualize this pattern
          if (propertyFilters.length > 0 && props.onVisualize) {
            console.log('Pattern clicked, extracted filters:', propertyFilters);
            props.onVisualize({
              active: true,
              propertyFilters,
              articleIds: [],
            });
          }
        }
      }
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  // If parent controls visibility, honor it
  const visible = () => props.isOpen === undefined ? true : props.isOpen;

  return (
    <>
      {/* Chat Interface - Controlled visibility */}
      <Show when={visible()}>
        <div class={`chat-interface ${isMinimized() ? 'minimized' : ''}`}>
          <div class="chat-topbar">
            <div class="chat-identity">
              <span class="chat-orb"><AssistantGlyph /></span>
              <div>
                <div class="chat-title">Pattern Companion</div>
                <div class="chat-kicker">PeTTa reasoning</div>
              </div>
            </div>
            <button class="chat-action-btn" onClick={clearChat} title="Clear chat" aria-label="Clear chat">
              <ClearIcon />
            </button>
          </div>

          <Show when={!isMinimized()}>
            <div class="chat-messages" ref={chatContainerRef}>
              <Show when={messages().length === 0}>
                <div class="chat-welcome">
                  <div class="welcome-icon"><AssistantGlyph /></div>
                  <h4>No conversation yet</h4>
                  <div class="welcome-suggestions">
                    <button class="suggestion-btn" onClick={() => {
                      setInputText('Why does article A_14219 got low engagement?');
                      inputRef?.focus();
                    }}>
                      Explain article A_14219
                    </button>
                    <button class="suggestion-btn" onClick={() => {
                      setInputText('Mine rules with 5 conjuncts and support 3');
                      inputRef?.focus();
                    }}>
                      Mine stronger rules
                    </button>
                    <button class="suggestion-btn" onClick={() => {
                      setInputText('Summarize the mined rules and tell me which are most actionable');
                      inputRef?.focus();
                    }}>
                      Summarize actionable rules
                    </button>
                  </div>
                </div>
              </Show>

              <For each={messages()}>
                {(message) => (
                  message.role === 'user' ? (
                    <div class={`message user`}>
                      <div class="message-content">
                        <div class="message-text" innerHTML={formatMessage(message.content)} />
                        <div class="message-time">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  ) : message.role === 'assistant' && message.isTyping ? (
                    <div class="message assistant">
                      <div class="message-avatar"><AssistantGlyph /></div>
                      <div class="message-content">
                        <div class="typing-indicator">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      </div>
                    </div>
                  ) : message.role === 'assistant' && !message.conjunct ? (
                    <div class="message assistant">
                      <div class="message-avatar"><AssistantGlyph /></div>
                      <div class="message-content">
                        <div class="message-text" innerHTML={formatMessage(message.content)} onClick={(e) => handlePatternClick(e, message)} />
                        <Show when={summarizeFunctionCalls(message.functionCalls)}>
                          {(trace) => <div class="function-trace">{trace()}</div>}
                        </Show>
                        <div class="message-time">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  ) : message.role === 'system' ? (
                    <div class="message system">
                      <div class="message-content">
                        <div class="message-text">{message.content}</div>
                      </div>
                    </div>
                  ) : null
                )}
              </For>
            </div>

            <div class="chat-input-container">
              <textarea
                ref={inputRef}
                class="chat-input"
                placeholder="Ask for a proof, rule summary, or mining run..."
                value={inputText()}
                onInput={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading()}
                rows={1}
              />
              <button
                class="send-btn"
                onClick={sendMessage}
                disabled={!inputText().trim() || isLoading()}
                aria-label="Send message"
              >
                <Show when={isLoading()} fallback={<SendIcon />}>
                  <span class="send-loading" />
                </Show>
              </button>
            </div>
          </Show>
        </div>
      </Show>
    </>
  );
};

// Helper function to format message content with markdown-like syntax and pattern references
const formatMessage = (content: string): string => {
  const codeBlocks: string[] = [];
  let formatted = content
    .replace(/```(?:\w+)?\n?([\s\S]*?)```/g, (_match, code) => {
      const token = `@@CODE_BLOCK_${codeBlocks.length}@@`;
      codeBlocks.push(`<pre><code>${String(code).trim()}</code></pre>`);
      return token;
    })
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  codeBlocks.forEach((block, index) => {
    formatted = formatted.replace(`@@CODE_BLOCK_${index}@@`, block);
  });

  // Convert [Pattern N] or [N] to clickable references
  formatted = formatted.replace(/\[(?:Rule )?(\d+)\]/g, '<span class="pattern-ref" data-pattern="$1" title="Click to visualize this pattern">[$1]</span>');

  formatted = formatted.replace(/\n/g, '<br/>');

  return formatted;
};

export default ChatInterface;
