import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import InputBar from './components/InputBar';

function App() {
    const [sessions, setSessions] = useState([]);
    const [currentSessionId, setCurrentSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [useAgent, setUseAgent] = useState(false);
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

    // --- Session Logic ---
    const fetchSessions = async () => {
        try {
            const res = await fetch('/v1/sessions');
            if (res.ok) {
                const data = await res.json();
                setSessions(Array.isArray(data) ? data : []);
            }
        } catch (err) {
            console.error("Failed to fetch sessions", err);
        }
    };

    const createSession = async () => {
        try {
            const res = await fetch('/v1/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: "New Chat" })
            });
            if (res.ok) {
                const session = await res.json();
                setSessions(prev => [session, ...prev]);
                setCurrentSessionId(session.id);
                setMessages([]);
            }
        } catch (err) {
            console.error("Failed to create session", err);
        }
    };

    const deleteSession = async (id) => {
        try {
            await fetch(`/v1/sessions/${id}`, { method: 'DELETE' });
            setSessions(prev => prev.filter(s => s.id !== id));
            if (currentSessionId === id) {
                setCurrentSessionId(null);
                setMessages([]);
            }
        } catch (err) {
            console.error("Failed to delete session", err);
        }
    };

    const selectSession = async (id) => {
        setCurrentSessionId(id);
        try {
            const res = await fetch(`/v1/sessions/${id}/messages`);
            if (res.ok) {
                const data = await res.json();
                setMessages(Array.isArray(data) ? data : []);
            }
        } catch (err) {
            console.error("Failed to fetch messages", err);
            setMessages([]);
        }
    };

    useEffect(() => {
        fetchSessions();
    }, []);

    // --- Send Logic ---
    const handleSendMessage = async (textOverride) => {
        const textToSend = (typeof textOverride === 'string') ? textOverride : inputText;
        if (!textToSend.trim()) return;

        let activeSessionId = currentSessionId;
        if (!activeSessionId) {
            try {
                const res = await fetch('/v1/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: textToSend.slice(0, 30) || "New Chat" })
                });
                const session = await res.json();
                setSessions(prev => [session, ...prev]);
                setCurrentSessionId(session.id);
                activeSessionId = session.id;
            } catch (e) { return; }
        }

        const userMsg = { role: 'user', content: textToSend };
        setMessages(prev => [...prev, userMsg]);
        setInputText("");
        setIsLoading(true);

        try {
            const endpoint = useAgent ? '/v1/agent/run' : '/v1/chat/completions';
            const body = useAgent
                ? { goal: userMsg.content, session_id: activeSessionId }
                : { messages: [userMsg], model: 'local-model', session_id: activeSessionId };

            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!res.ok) throw new Error("Failed");

            const data = await res.json();
            const content = useAgent ? data.answer : data.choices[0].message.content;

            setMessages(prev => [...prev, { role: 'assistant', content }]);
            fetchSessions(); // Refresh list for titles
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: "Error: " + err.message }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleAudioRecorded = async (blob) => {
        setIsLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');
            const res = await fetch('/v1/audio/transcriptions', { method: 'POST', body: formData });
            if (!res.ok) throw new Error("Audio failed");
            const data = await res.json();
            // Automatically send the transcibed text
            await handleSendMessage(data.text);
        } catch (err) {
            alert("Transcription failed");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-white text-[#1f1f1f] font-sans overflow-hidden">
            <Sidebar
                sessions={sessions}
                currentSessionId={currentSessionId}
                onNewChat={createSession}
                onSelectSession={selectSession}
                onDeleteSession={deleteSession}
                isCollapsed={isSidebarCollapsed}
                toggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            />

            <main className="flex-1 flex flex-col h-full bg-white relative">
                <div className="flex-1 overflow-hidden relative">
                    <ChatArea messages={messages} isStreaming={isLoading} />
                </div>

                <InputBar
                    inputText={inputText}
                    setInputText={setInputText}
                    onSend={handleSendMessage}
                    isLoading={isLoading}
                    useAgent={useAgent}
                    setUseAgent={setUseAgent}
                    onAudioRecorded={handleAudioRecorded}
                />
            </main>
        </div>
    );
}

export default App;
