import React, { useEffect, useRef, memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Copy, Check, Terminal } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const MessageItem = memo(({ msg }) => {
    return (
        <div className={`group flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''} mb-6 animate-fade-in`}>
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 ${msg.role === 'user' ? 'bg-[#c2e7ff] text-[#001d35]' : 'bg-gradient-to-tr from-[#4285f4] to-[#9b72cb] text-white'
                }`}>
                {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
            </div>

            {/* Content */}
            <div className={`max-w-[85%] text-[15px] leading-relaxed text-[#1f1f1f] ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                {msg.role === 'user' ? (
                    <div className="bg-[#f0f4f9] px-5 py-3 rounded-[20px] rounded-tr-sm inline-block text-left">
                        {msg.content}
                    </div>
                ) : (
                    <div className="prose prose-slate max-w-none">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '')
                                    return !inline && match ? (
                                        <div className="rounded-md overflow-hidden my-2 border border-blue-100 bg-white">
                                            <div className="bg-[#f8f9fa] px-3 py-1 flex items-center justify-between border-b border-gray-100">
                                                <span className="text-xs font-mono text-gray-500">{match[1]}</span>
                                            </div>
                                            <SyntaxHighlighter
                                                style={oneLight}
                                                language={match[1]}
                                                PreTag="div"
                                                customStyle={{ margin: 0, padding: '1rem', fontSize: '0.9em' }}
                                                {...props}
                                            >
                                                {String(children).replace(/\n$/, '')}
                                            </SyntaxHighlighter>
                                        </div>
                                    ) : (
                                        <code className={`${className} bg-gray-100 px-1 py-0.5 rounded text-sm font-mono text-pink-600`} {...props}>
                                            {children}
                                        </code>
                                    )
                                }
                            }}
                        >
                            {msg.content}
                        </ReactMarkdown>
                    </div>
                )}

                {/* Related Memories */}
                {msg.relatedMemories && msg.relatedMemories.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-100 flex flex-col gap-2">
                        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                            <Terminal size={12} /> Related Memories
                        </div>
                        <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
                            {msg.relatedMemories.map(mem => (
                                <div key={mem.id} className="min-w-[200px] w-[200px] bg-white border border-gray-200 p-2 rounded text-xs text-gray-600 shadow-sm">
                                    <p className="line-clamp-2 mb-1">{mem.content}</p>
                                    <span className="text-[10px] text-blue-500">{(mem.score * 100).toFixed(0)}% relevance</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
});

const ChatArea = ({ messages, isStreaming }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isStreaming]);

    if (!messages || messages.length === 0) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <div className="mb-4">
                    <span className="text-5xl bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570] bg-clip-text text-transparent font-medium tracking-tight">
                        Hello, Artemis
                    </span>
                </div>
                <p className="text-2xl text-[#444746] font-normal mb-8">How can I help you today?</p>

                {/* Suggestions Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl w-full">
                    <div className="bg-[#f0f4f9] hover:bg-[#dfe3e8] p-4 rounded-xl cursor-pointer text-left transition-colors">
                        <p className="text-sm font-medium text-[#1f1f1f] mb-1">Create an image</p>
                        <p className="text-xs text-[#444746]">of a futuristic coding environment</p>
                    </div>
                    <div className="bg-[#f0f4f9] hover:bg-[#dfe3e8] p-4 rounded-xl cursor-pointer text-left transition-colors">
                        <p className="text-sm font-medium text-[#1f1f1f] mb-1">Help me code</p>
                        <p className="text-xs text-[#444746]">a Python script for data analysis</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full overflow-y-auto px-4 py-8 custom-scrollbar">
            <div className="max-w-[800px] mx-auto">
                {messages.map((msg, i) => <MessageItem key={i} msg={msg} />)}
                {isStreaming && (
                    <div className="flex items-center gap-2 mb-4 animate-pulse ml-12">
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 delay-75"></div>
                        <div className="w-2 h-2 rounded-full bg-blue-500 delay-150"></div>
                    </div>
                )}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default ChatArea;
