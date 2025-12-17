import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import AudioRecorder from './components/AudioRecorder';
import AgentStatus from './components/AgentStatus';
import { Send, Bot, FileText } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [useAgent, setUseAgent] = useState(false);
  const [agentStatus, setAgentStatus] = useState(null); // 'thinking', 'searching', 'transcribing'

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMsg = { role: 'user', content: inputText };
    setMessages(prev => [...prev, userMsg]);
    setInputText("");
    setIsLoading(true);

    try {
      if (useAgent) {
        setAgentStatus('searching');
        const res = await fetch('/v1/agent/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: userMsg.content }),
        });

        if (!res.ok) throw new Error("Agent failed");

        const data = await res.json();
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
      } else {
        setAgentStatus('thinking');
        // Simple chat completion
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [...messages, userMsg], // Send history
            model: 'local-model',
            temperature: 0.7
          }),
        });

        if (!res.ok) throw new Error("Chat failed");

        const data = await res.json();
        const content = data.choices[0].message.content;
        setMessages(prev => [...prev, { role: 'assistant', content }]);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setIsLoading(false);
      setAgentStatus(null);
    }
  };

  const handleAudioRecorded = async (blob) => {
    setAgentStatus('transcribing');
    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.webm');

      const res = await fetch('/v1/audio/transcriptions', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error("Transcription failed");

      const data = await res.json();
      setInputText(prev => (prev ? `${prev} ${data.text}` : data.text));
    } catch (err) {
      console.error(err);
      alert("Transcription failed");
    } finally {
      setAgentStatus(null);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-neutral-900 text-gray-100 font-sans">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-neutral-800 border-b border-neutral-700 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-900/50">
            <Bot className="text-white" size={24} />
          </div>
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Artemis
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <AgentStatus status={agentStatus} />
          <div className="flex items-center gap-2 bg-neutral-900 p-1 rounded-lg border border-neutral-700">
            <button
              onClick={() => setUseAgent(false)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${!useAgent ? 'bg-neutral-700 text-white shadow' : 'text-gray-400 hover:text-white'}`}
            >
              Chat
            </button>
            <button
              onClick={() => setUseAgent(true)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${useAgent ? 'bg-blue-900/50 text-blue-100 shadow border border-blue-800' : 'text-gray-400 hover:text-white'}`}
            >
              Agent
            </button>
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 overflow-hidden relative flex flex-col">
        <ChatWindow messages={messages} />
      </main>

      {/* Input Area */}
      <footer className="p-4 bg-neutral-800 border-t border-neutral-700">
        <div className="max-w-4xl mx-auto flex items-end gap-3">
          <AudioRecorder onAudioRecorded={handleAudioRecorded} disabled={isLoading} />

          <div className="flex-1 bg-neutral-900 rounded-2xl border border-neutral-700 focus-within:ring-2 focus-within:ring-blue-600/50 transition-all flex items-center p-2">
            <textarea
              className="flex-1 bg-transparent border-none text-white placeholder-gray-500 focus:ring-0 resize-none max-h-32 py-2 px-2"
              rows={1}
              placeholder={useAgent ? "Ask Artemis to do something..." : "Type a message..."}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={isLoading}
            />
          </div>

          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
            className="p-3 bg-blue-600 hover:bg-blue-500 text-white rounded-full transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-blue-900/20"
          >
            <Send size={24} />
          </button>
        </div>
        <div className="text-center mt-2 text-xs text-neutral-500">
          Hardware Accelerated • Local Inference • Private
        </div>
      </footer>
    </div>
  );
}

export default App;
