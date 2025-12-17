import React, { useRef, useEffect } from 'react';
import { User, Bot } from 'lucide-react';

const ChatWindow = ({ messages }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                    <Bot size={48} className="mb-2 opacity-50" />
                    <p>Start a conversation with Artemis</p>
                </div>
            )}

            {messages.map((msg, idx) => (
                <div
                    key={idx}
                    className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                    {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center flex-shrink-0">
                            <Bot size={18} className="text-white" />
                        </div>
                    )}

                    <div
                        className={`max-w-[80%] p-3 rounded-2xl whitespace-pre-wrap ${msg.role === 'user'
                                ? 'bg-blue-600 text-white rounded-tr-none'
                                : 'bg-gray-700 text-gray-100 rounded-tl-none'
                            }`}
                    >
                        {msg.content}
                    </div>

                    {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-blue-800 flex items-center justify-center flex-shrink-0">
                            <User size={18} className="text-white" />
                        </div>
                    )}
                </div>
            ))}
            <div ref={bottomRef} />
        </div>
    );
};

export default ChatWindow;
