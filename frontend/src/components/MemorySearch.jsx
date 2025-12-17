
import React, { useState } from 'react';

// Simplified Memory Search Component
export default function MemorySearch({ onSelectMemory }) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        try {
            const response = await fetch('/v1/memory/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, top_k: 10 }),
            });

            if (response.ok) {
                const data = await response.json();
                setResults(data);
            }
        } catch (error) {
            console.error("Search failed:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleArchiveStats = async () => {
        // Demo function to show stats if needed
        // In real implementation could fetch /v1/memory/archive/stats
    };

    return (
        <div className="flex flex-col h-full bg-gray-900 text-white p-4">
            <h2 className="text-xl font-bold mb-4">Memory Bank</h2>

            <form onSubmit={handleSearch} className="mb-4">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search memories..."
                        className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                    />
                    <button
                        type="submit"
                        disabled={loading}
                        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm disabled:opacity-50"
                    >
                        {loading ? '...' : 'Search'}
                    </button>
                </div>
            </form>

            <div className="flex-1 overflow-y-auto space-y-3">
                {results.length === 0 && !loading && (
                    <p className="text-gray-500 text-center text-sm mt-10">No memories found</p>
                )}

                {results.map((memory) => (
                    <div
                        key={memory.id}
                        className="bg-gray-800 p-3 rounded hover:bg-gray-750 cursor-pointer border border-gray-700"
                        onClick={() => onSelectMemory && onSelectMemory(memory)}
                    >
                        <p className="text-sm text-gray-300 line-clamp-3">{memory.content}</p>
                        <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
                            <span className="capitalize">{memory.role || 'unknown'}</span>
                            <span>{new Date(memory.created_at).toLocaleDateString()}</span>
                            <span>{(memory.score * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-4 pt-4 border-t border-gray-800 flex justify-between text-xs text-gray-500">
                <button className="hover:text-blue-400">Export All</button>
                <button className="hover:text-blue-400">Cleanup</button>
            </div>
        </div>
    );
}
