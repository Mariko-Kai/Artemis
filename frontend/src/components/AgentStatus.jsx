import React from 'react';
import { Loader2, BrainCircuit } from 'lucide-react';

const AgentStatus = ({ status }) => {
    if (!status) return null;

    return (
        <div className="flex items-center gap-2 text-sm text-blue-400 bg-blue-900/20 px-3 py-1 rounded-full animate-fade-in">
            {status === 'thinking' && <BrainCircuit size={16} className="animate-pulse" />}
            {status === 'searching' && <Loader2 size={16} className="animate-spin" />}
            <span>
                {status === 'thinking' && "Agent is thinking..."}
                {status === 'searching' && "Agent is searching..."}
                {status === 'transcribing' && "Transcribing audio..."}
            </span>
        </div>
    );
};

export default AgentStatus;
