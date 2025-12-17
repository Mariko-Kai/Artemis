import React, { useState } from 'react';
import { Menu, Plus, MessageSquare, Settings, ChevronLeft, ChevronRight, History, Trash2 } from 'lucide-react';
import { format } from 'date-fns';

const Sidebar = ({ sessions, currentSessionId, onNewChat, onSelectSession, onDeleteSession, isCollapsed, toggleSidebar }) => {
    return (
        <aside className={`bg-[#f0f4f9] h-full flex flex-col transition-all duration-300 ${isCollapsed ? 'w-16 items-center' : 'w-[280px]'}`}>

            {/* Top Section: Hamburger & New Chat */}
            <div className="p-4 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                    <button onClick={toggleSidebar} className="p-2 hover:bg-gray-200 rounded-full transition-colors text-gray-600">
                        <Menu size={24} />
                    </button>
                </div>

                <button
                    onClick={onNewChat}
                    className={`flex items-center gap-3 bg-[#dde3ea] hover:bg-[#d0d7de] text-[#1f1f1f] rounded-xl transition-all ${isCollapsed ? 'p-3 justify-center' : 'px-4 py-3'}`}
                    title="New chat"
                >
                    <Plus size={20} className="text-gray-600" />
                    {!isCollapsed && <span className="font-medium text-sm">New chat</span>}
                </button>
            </div>

            {/* Recent List */}
            <div className="flex-1 overflow-y-auto px-2 mt-2">
                {!isCollapsed && <div className="px-4 mb-2 text-sm font-medium text-gray-600">Recent</div>}

                <div className="space-y-1">
                    {sessions.map((session) => (
                        <div
                            key={session.id}
                            className={`w-full group flex items-center gap-3 px-3 py-2 rounded-full transition-colors text-left cursor-pointer ${currentSessionId === session.id
                                ? 'bg-[#d3e3fd] text-[#001d35]'
                                : 'hover:bg-[#e6e9ef] text-[#1f1f1f]'
                                }`}
                            onClick={() => onSelectSession(session.id)}
                        >
                            <MessageSquare size={18} className="flex-shrink-0" />
                            {!isCollapsed && (
                                <>
                                    <span className="text-sm truncate flex-1">{session.title || "New conversation"}</span>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDeleteSession(session.id);
                                        }}
                                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 text-gray-500 hover:text-red-500 rounded transition-all"
                                        title="Delete chat"
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </>
                            )}
                        </div>
                    ))}
                    {sessions.length === 0 && !isCollapsed && (
                        <div className="px-4 text-xs text-gray-500">No recent chats</div>
                    )}
                </div>
            </div>

            {/* Bottom Section: Settings */}
            <div className="p-2 mt-auto">
                <button className={`w-full flex items-center gap-3 px-3 py-2 rounded-full hover:bg-[#e6e9ef] text-[#1f1f1f] transition-colors ${isCollapsed ? 'justify-center' : ''}`}>
                    <Settings size={20} />
                    {!isCollapsed && <span className="text-sm">Settings</span>}
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;
