import React, { useRef, useEffect } from 'react';
import { Mic, Send, Image as ImageIcon, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import AudioRecorder from './AudioRecorder';

const InputBar = ({
    inputText,
    setInputText,
    onSend,
    isLoading,
    useAgent,
    setUseAgent,
    onAudioRecorded
}) => {
    const textareaRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
        }
    }, [inputText]);

    return (
        <div className="max-w-[800px] w-full mx-auto px-4 pb-4">
            <div className={clsx(
                "bg-[#f0f4f9] rounded-[28px] overflow-hidden transition-all focus-within:bg-white focus-within:shadow-md border border-transparent focus-within:border-gray-200 relative",
                isLoading && "opacity-80 pointer-events-none"
            )}>

                {/* Input Area */}
                <div className="flex flex-col">
                    <textarea
                        ref={textareaRef}
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                onSend();
                            }
                        }}
                        placeholder={useAgent ? "Ask Artemis Agent..." : "Enter a prompt here"}
                        rows={1}
                        className="w-full bg-transparent border-none outline-none resize-none px-6 py-4 text-base text-gray-800 placeholder-gray-500 max-h-[150px] overflow-y-auto"
                        disabled={isLoading}
                    />
                </div>

                {/* Toolbar */}
                <div className="flex items-center justify-between px-4 pb-3">
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setUseAgent(!useAgent)}
                            className={clsx(
                                "p-2 rounded-full transition-colors flex items-center gap-2 text-sm font-medium",
                                useAgent ? "bg-[#c2e7ff] text-[#001d35]" : "hover:bg-gray-200 text-gray-600"
                            )}
                            title="Toggle Agent Mode (Tools)"
                        >
                            <Sparkles size={20} />
                            {useAgent && <span className="text-xs">Agent</span>}
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <AudioRecorder onAudioRecorded={onAudioRecorded} disabled={isLoading} />

                        {inputText.trim() && (
                            <button
                                onClick={onSend}
                                className="p-2 rounded-full hover:bg-gray-200 text-blue-600 transition-colors"
                            >
                                <Send size={20} />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            <div className="text-center text-xs text-gray-500 mt-2">
                Artemis may display inaccurate info, including about people, so double-check its responses.
            </div>
        </div>
    );
};

export default InputBar;
