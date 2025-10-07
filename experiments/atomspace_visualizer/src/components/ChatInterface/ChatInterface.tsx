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
  onVisualize: (conjunct: string) => void;
  miningResults?: Array<{ pattern: string; support: string }>;
  conjunctSize?: number;
  onMiningStart?: (conjunctSize: number) => void;
}

const ChatInterface = (props: ChatInterfaceProps) => {
  const [messages, setMessages] = createSignal<Message[]>([]);
  const [inputText, setInputText] = createSignal('');
  const [isLoading, setIsLoading] = createSignal(false);
  const [isMinimized, setIsMinimized] = createSignal(false);
  const [isChatOpen, setIsChatOpen] = createSignal(false);
  const [lastConjunctSize, setLastConjunctSize] = createSignal<number | null>(null);

  let chatContainerRef: HTMLDivElement | undefined;
  let inputRef: HTMLTextAreaElement | undefined;
  
  // Auto-send mining message when conjunct size changes
  createEffect(() => {
    const conjunctSize = props.conjunctSize;
    console.log('ChatInterface: conjunctSize changed to:', conjunctSize, 'last:', lastConjunctSize());
    if (conjunctSize && conjunctSize !== lastConjunctSize()) {
      console.log('ChatInterface: Opening chat and sending message');
      setLastConjunctSize(conjunctSize);
      setIsChatOpen(true);
      
      // Auto-send user message
      const userMessage = `Mine rules with ${conjunctSize} patterns`;
      const userMsg: Message = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'user',
        content: userMessage,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMsg]);
      
      // Get AI response
      setTimeout(() => {
        sendAIMessage(userMessage);
      }, 300);
    }
  });

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
      setIsChatOpen(true);
      
      // Add system message about mining completion
      const systemMsg: Message = {
        id: `sys-${Date.now()}`,
        role: 'system',
        content: `Mining completed! Found ${results.length} pattern(s). Analyzing results...`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, systemMsg]);
      
      // Process each result with AI
      setTimeout(() => {
        results.forEach((result, index) => {
          setTimeout(() => {
            analyzeConjunct(result.pattern, result.support);
          }, index * 1000);
        });
      }, 500);
    }
  });

  const analyzeConjunct = async (pattern: string, support: string) => {
    try {
      const response = await fetch('http://localhost:5000/api/chat/analyze', {
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
      const response = await fetch('http://localhost:5000/api/chat', {
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
    
    // Get AI response
    await sendAIMessage(text);
  };

  const handleFunctionCall = (functionCall: any) => {
    // Handle various function calls from the AI
    console.log('Function call:', functionCall);
  };

  const handleVisualize = (pattern: string) => {
    props.onVisualize(pattern);
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen());
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <>
      {/* Floating Chat Button */}
      <Show when={!isChatOpen()}>
        <button class="chat-toggle-btn" onClick={toggleChat}>
          💬
          <Show when={messages().length > 0}>
            <span class="chat-badge">{messages().length}</span>
          </Show>
        </button>
      </Show>

      {/* Chat Interface */}
      <Show when={isChatOpen()}>
        <div class={`chat-interface ${isMinimized() ? 'minimized' : ''}`}>
          <div class="chat-header">
            <div class="chat-header-left">
              <span class="chat-icon">🤖</span>
              <div class="chat-title-info">
                <h3>AI Assistant</h3>
                <span class="chat-status">Online</span>
              </div>
            </div>
            <div class="chat-header-actions">
              <button class="chat-action-btn" onClick={clearChat} title="Clear Chat">
                🗑️
              </button>
              <button class="chat-action-btn" onClick={() => setIsMinimized(!isMinimized())} title="Minimize">
                {isMinimized() ? '□' : '−'}
              </button>
              <button class="chat-action-btn" onClick={toggleChat} title="Close">
                ×
              </button>
            </div>
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
                  <div class={`message ${message.role}`}>
                    <Show when={message.role === 'assistant' && !message.isTyping}>
                      <div class="message-avatar">🤖</div>
                    </Show>
                    <Show when={message.role === 'user'}>
                      <div class="message-avatar">👤</div>
                    </Show>
                    
                    <div class="message-content">
                      <Show when={message.isTyping}>
                        <div class="typing-indicator">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      </Show>
                      
                      <Show when={!message.isTyping}>
                        <div class="message-text" innerHTML={formatMessage(message.content)} />
                        
                        <Show when={message.conjunct}>
                          <div class="conjunct-card">
                            <div class="conjunct-header">
                              <span class="conjunct-label">Pattern</span>
                              <span class="conjunct-support">Support: {message.conjunct!.support}</span>
                            </div>
                            <pre class="conjunct-pattern">{message.conjunct!.pattern}</pre>
                            <button 
                              class="visualize-btn"
                              onClick={() => handleVisualize(message.conjunct!.pattern)}
                            >
                              👁️ Visualize
                            </button>
                          </div>
                        </Show>
                        
                        <div class="message-time">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </Show>
                    </div>
                  </div>
                )}
              </For>
            </div>

            <div class="chat-input-container">
              <textarea
                ref={inputRef}
                class="chat-input"
                placeholder="Ask me anything about the patterns..."
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
                {isLoading() ? '⏳' : '📤'}
              </button>
            </div>
          </Show>
        </div>
      </Show>
    </>
  );
};

// Helper function to format message content with markdown-like syntax
const formatMessage = (content: string): string => {
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/```([^```]+)```/g, '<pre><code>$1</code></pre>')
    .replace(/\n/g, '<br/>');
};

export default ChatInterface;
