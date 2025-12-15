import { createSignal, For, Show, onMount, createEffect } from 'solid-js';
import './ChatInterface.css';

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
  isTyping?: boolean;
}

export interface ChatInterfaceProps {
  onVisualize: (filter: import('../../types').FilterState) => void;
  miningResults?: Array<{ pattern: string; support: string }>;
  conjunctSize?: number;
  onMiningStart?: (conjunctSize: number) => void;
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
        const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
        try {
          const resp = await fetch(`${API_BASE}/api/chat/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patterns: results })
          });
          if (resp.ok) {
            const data = await resp.json();
            const assistantMsg: Message = {
              id: `msg-${Date.now()}-${Math.random()}`,
              role: 'assistant',
              content: data.summary,
              timestamp: new Date()
            };
            setMessages(prev => [...prev, assistantMsg]);
          } else {
            // Fallback: analyze each conjunct individually
            results.forEach((result, index) => {
              setTimeout(() => {
                analyzeConjunct(result.pattern, result.support);
              }, index * 500);
            });
          }
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
    const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
    
    try {
      const response = await fetch(`${API_BASE}/api/chat/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pattern, support })
      });

      if (!response.ok) throw new Error('Failed to analyze conjunct');

      const data = await response.json();
      
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
        content: `📊 **Pattern Found** (Support: ${support})\n\nThis pattern shows a relationship between topics and their properties:\n\`\`\`\n${pattern}\n\`\`\`\n\nThis pattern appears ${support} times in the dataset.`,
        timestamp: new Date(),
        conjunct: { pattern, support }
      };

      setMessages(prev => [...prev, fallbackMsg]);
    }
  };

  // Send AI message (internal function)
  const sendAIMessage = async (text: string) => {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
    
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
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages().filter(m => !m.isTyping && m.role !== 'system').map(m => ({
            role: m.role,
            content: m.content
          })),
          session_id: 'default'
        })
      });

      if (!response.ok) throw new Error('Failed to get response');

      const data = await response.json();
      
      // Remove typing indicator and add actual response
      setMessages(prev => prev.filter(m => !m.isTyping));
      
      const aiMsg: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMsg]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => prev.filter(m => !m.isTyping));
      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: '❌ Sorry, I encountered an error. Please try again.',
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
    
    // Intercept explicit mining commands and delegate to parent unified miner
    const mineRegex = /mine(?: rules)?(?: with)?\s*(?:the )?(?:next )?(?:top )?\s*(\d+)\s*(?:patterns?|conjunctions?)/i;
    const m = text.match(mineRegex);
    if (m) {
      const n = parseInt(m[1], 10) || 5;
      // If parent provides the unified mining handler, use it and avoid
      // sending the message to the AI (which may also call the mining
      // function and duplicate work).
      if (props.onMiningStart) {
        const sys: Message = {
          id: `sys-start-${Date.now()}`,
          role: 'system',
          content: `Starting mining with ${n} patterns...`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, sys]);
        try {
          props.onMiningStart(n);
        } catch (err) {
          console.error('Error delegating mining to parent from chat:', err);
        }
        return;
      }
      // fallback: no parent handler -> ask AI to perform mining intent
    }

    // Get AI response
    await sendAIMessage(text);
  };

  const handleFunctionCall = (functionCall: any) => {
    // Handle various function calls from the AI
    console.log('Function call:', functionCall);
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
          // Example pattern: (length $x "low") (tone $x "Analytical")
          const regex = /\((\w+)\s+\$\w+\s+("|'|)([^"')]+)\2\)/g;
          const propertyFilters: { property: string; value: string }[] = [];
          let match;
          while ((match = regex.exec(patternObj.pattern)) !== null) {
            const property = match[1];
            const value = `"${ match[3]?.trim()}"`;
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
          {/* Header removed as it's now handled by the sidebar container */}
          <div class="chat-header-actions-floating">
             <button class="chat-action-btn" onClick={clearChat} title="Clear Chat">
                🗑️
              </button>
          </div>

          <Show when={!isMinimized()}>
            <div class="chat-container" ref={chatContainerRef}>
              <Show when={messages().length === 0}>
                <div class="chat-welcome">
                  <div class="welcome-icon">👋</div>
                  <h4>Welcome to AtomSpace AI Assistant!</h4>
                  <p>I can help you understand mining results, analyze patterns, and visualize data.</p>
                  <div class="welcome-suggestions">
                    <button class="suggestion-btn" onClick={() => {
                      setInputText('What patterns have been found?');
                      inputRef?.focus();
                    }}>
                      What patterns have been found?
                    </button>
                    <button class="suggestion-btn" onClick={() => {
                      setInputText('Explain the most common pattern');
                      inputRef?.focus();
                    }}>
                      Explain the most common pattern
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
                      <div class="message-avatar">🤖</div>
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
                      <div class="message-avatar">🤖</div>
                      <div class="message-content">
                        <div class="message-text" innerHTML={formatMessage(message.content)} onClick={(e) => handlePatternClick(e, message)} />
                        <div class="message-time">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
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
                placeholder="Ask me anything..."
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
              >
                {isLoading() ? '⏳' : '➤'}
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
  let formatted = content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/```([^```]+)```/g, '<pre><code>$1</code></pre>');
  
  // Convert [Pattern N] or [N] to clickable references
  formatted = formatted.replace(/\[(?:Rule )?(\d+)\]/g, '<span class="pattern-ref" data-pattern="$1" title="Click to visualize this pattern">[$1]</span>');
  
  formatted = formatted.replace(/\n/g, '<br/>');
  
  return formatted;
};

export default ChatInterface;
