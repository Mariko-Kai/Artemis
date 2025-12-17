
import React, { useState } from 'react';
import { Menu, MessageSquare, Plus, Settings, Search, Trash2, Database } from 'lucide-react';
import MemorySearch from './MemorySearch';

export default function Sidebar({ sessions, currentSessionId, onSelectSession, onNewSession, onDeleteSession }) {
    const [activeTab, setActiveTab] = useState('chats'); // 'chats' or 'memory'

    return (
        <div className="w-[280px] bg-[#f0f4f9] border-r border-gray-200 flex flex-col h-full flex-shrink-0">

            {/* Header Tabs */}
            <div className="flex border-b border-gray-200 bg-white">
                <button
                    className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${activeTab === 'chats' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    onClick={() => setActiveTab('chats')}
                >
                    <MessageSquare size={16} /> Chats
                </button>
                <button
                    className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${activeTab === 'memory' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    onClick={() => setActiveTab('memory')}
                >
                    <Database size={16} /> Memory
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden flex flex-col">
                {activeTab === 'chats' ? (
                    <>
                        {/* New Chat Button */}
                        <div className="p-4">
                            <button
                                onClick={onNewSession}
                                className="w-full bg-[#dde3ea] hover:bg-[#d0d7de] text-[#1f1f1f] rounded-xl py-3 px-4 flex items-center gap-3 transition-colors text-sm font-medium"
                            >
                                <Plus size={20} className="text-gray-600" />
                                <span>New chat</span>
                            </button>
                        </div>

                        {/* Chat List */}
                        <div className="flex-1 overflow-y-auto px-2">
                            <div className="px-4 mb-2 text-xs font-medium text-gray-500 uppercase tracking-wider">Recent</div>
                            <div className="space-y-1">
                                {sessions.map((session) => (
                                    <div
                                        key={session.id}
                                        className={`group w-full flex items-center gap-3 px-3 py-2 rounded-full cursor-pointer text-left transition-colors ${session.id === currentSessionId
                                                ? 'bg-[#d3e3fd] text-[#001d35]'
                                                : 'hover:bg-[#e6e9ef] text-[#1f1f1f]'
                                            }`}
                                        onClick={() => onSelectSession(session.id)}
                                    >
                                        <MessageSquare size={16} className="flex-shrink-0 opacity-70" />
                                        <span className="text-sm truncate flex-1">{session.title || 'New conversation'}</span>

                                        {onDeleteSession && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onDeleteSession(session.id);
                                                }}
                                                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-100 text-gray-400 hover:text-red-500 rounded-full transition-all"
                                                title="Delete chat"
                                            >
                                                <Trash2 size={12} />
                                            </button>
                                        )}
                                    </div>
                                ))}
                                {sessions.length === 0 && (
                                    <p className="px-4 text-xs text-gray-400 italic mt-2">No chats yet</p>
                                )}
                            </div>
                        </div>

                        {/* Bottom Settings */}
                        <div className="p-2 border-t border-gray-200 mt-auto">
                            <button className="w-full flex items-center gap-3 px-3 py-2 rounded-full hover:bg-[#e6e9ef] text-[#1f1f1f] transition-colors">
                                <Settings size={18} />
                                <span className="text-sm">Settings</span>
                            </button>
                        </div>
                    </>
                ) : (
                    <MemorySearch />
                )}
            </div>
        </div>
    );
}
