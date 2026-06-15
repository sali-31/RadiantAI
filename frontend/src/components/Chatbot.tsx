import React, { useState, useRef, useEffect, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { ProductCard } from './RecommendedProducts';
import type { Product } from '../types';

interface Message {
    id: string;
    type: 'user' | 'bot';
    content: string;
    timestamp: Date;
    products?: Product[];
}

interface Props {
    userId: string;
}

interface StoredMessage extends Omit<Message, 'timestamp'> {
    timestamp: string;
}

interface ChatThread {
    id: string;
    title: string;
    createdAt: Date;
    updatedAt: Date;
    messages: Message[];
}

interface StoredChatThread extends Omit<ChatThread, 'createdAt' | 'updatedAt' | 'messages'> {
    createdAt: string;
    updatedAt: string;
    messages: StoredMessage[];
}

interface SkinProfile {
    skin_type: string | null;
    concerns: string[];
    goals: string[];
    preferences: string[];
    avoid: string[];
}

interface MemoryUpdates {
    skin_type?: string | null;
    concerns?: string[];
    goals?: string[];
    preferences?: string[];
    avoid?: string[];
}

const createWelcomeMessage = (): Message => ({
    id: 'welcome',
    type: 'bot',
    content: 'Hello! 👋 I\'m your AI skin health assistant. Ask me anything about skin conditions, treatments, skincare routines, or general dermatology advice. What would you like to know?',
    timestamp: new Date(),
});

const getLegacyStorageKey = (userId: string) => `radiantai_chat_messages_${userId || 'anonymous'}`;
const getThreadsStorageKey = (userId: string) => `radiantai_chat_threads_${userId || 'anonymous'}`;
const getActiveThreadStorageKey = (userId: string) => `radiantai_active_chat_thread_${userId || 'anonymous'}`;
const getSkinProfileStorageKey = (userId: string) => `radiantai_skin_profile_${userId || 'anonymous'}`;

const defaultSkinProfile: SkinProfile = {
    skin_type: null,
    concerns: [],
    goals: [],
    preferences: [],
    avoid: [],
};

const createEmptyThread = (): ChatThread => {
    const now = new Date();
    return {
        id: `thread-${now.getTime()}`,
        title: 'New chat',
        createdAt: now,
        updatedAt: now,
        messages: [createWelcomeMessage()],
    };
};

const deserializeMessages = (messages: StoredMessage[]): Message[] => {
    return messages.map(message => ({
        ...message,
        timestamp: new Date(message.timestamp),
    }));
};

const serializeMessages = (messages: Message[]): StoredMessage[] => {
    return messages.map(message => ({
        ...message,
        timestamp: message.timestamp.toISOString(),
    }));
};

const deriveThreadTitle = (messages: Message[]) => {
    const firstUserMessage = messages.find(message => message.type === 'user' && message.content.trim());
    if (!firstUserMessage) return 'New chat';
    const compact = firstUserMessage.content.trim().replace(/\s+/g, ' ');
    return compact.length > 42 ? `${compact.slice(0, 42)}...` : compact;
};

const serializeThreads = (threads: ChatThread[]): StoredChatThread[] => {
    return threads.map(thread => ({
        ...thread,
        createdAt: thread.createdAt.toISOString(),
        updatedAt: thread.updatedAt.toISOString(),
        messages: serializeMessages(thread.messages),
    }));
};

const deserializeThreads = (threads: StoredChatThread[]): ChatThread[] => {
    return threads.map(thread => ({
        ...thread,
        createdAt: new Date(thread.createdAt),
        updatedAt: new Date(thread.updatedAt),
        messages: deserializeMessages(thread.messages),
    }));
};

const loadThreads = (userId: string): ChatThread[] => {
    try {
        const savedThreads = localStorage.getItem(getThreadsStorageKey(userId));
        if (savedThreads) {
            const parsedThreads = JSON.parse(savedThreads) as StoredChatThread[];
            if (Array.isArray(parsedThreads) && parsedThreads.length > 0) {
                return deserializeThreads(parsedThreads);
            }
        }

        const legacySaved = localStorage.getItem(getLegacyStorageKey(userId));
        if (legacySaved) {
            const legacyMessages = JSON.parse(legacySaved) as StoredMessage[];
            if (Array.isArray(legacyMessages) && legacyMessages.length > 0) {
                const messages = deserializeMessages(legacyMessages);
                const now = new Date();
                return [{
                    id: `thread-${now.getTime()}`,
                    title: deriveThreadTitle(messages),
                    createdAt: now,
                    updatedAt: now,
                    messages,
                }];
            }
        }

        return [createEmptyThread()];
    } catch (error) {
        console.error('Failed to load chat history:', error);
        return [createEmptyThread()];
    }
};

const getInitialActiveThreadId = (userId: string, threads: ChatThread[]) => {
    const saved = localStorage.getItem(getActiveThreadStorageKey(userId));
    if (saved && threads.some(thread => thread.id === saved)) return saved;
    return threads[0].id;
};

const uniqueMerge = (current: string[], incoming?: string[]) => {
    const next = [...current];
    (incoming || []).forEach(value => {
        const normalized = String(value || '').trim();
        if (normalized && !next.includes(normalized)) {
            next.push(normalized);
        }
    });
    return next;
};

const loadSkinProfile = (userId: string): SkinProfile => {
    try {
        const saved = localStorage.getItem(getSkinProfileStorageKey(userId));
        if (!saved) return defaultSkinProfile;
        const parsed = JSON.parse(saved) as Partial<SkinProfile>;
        return {
            skin_type: parsed.skin_type || null,
            concerns: Array.isArray(parsed.concerns) ? parsed.concerns : [],
            goals: Array.isArray(parsed.goals) ? parsed.goals : [],
            preferences: Array.isArray(parsed.preferences) ? parsed.preferences : [],
            avoid: Array.isArray(parsed.avoid) ? parsed.avoid : [],
        };
    } catch (error) {
        console.error('Failed to load skin profile:', error);
        return defaultSkinProfile;
    }
};

const mergeSkinProfile = (profile: SkinProfile, updates?: MemoryUpdates): SkinProfile => {
    if (!updates) return profile;
    return {
        skin_type: updates.skin_type || profile.skin_type,
        concerns: uniqueMerge(profile.concerns, updates.concerns),
        goals: uniqueMerge(profile.goals, updates.goals),
        preferences: uniqueMerge(profile.preferences, updates.preferences),
        avoid: uniqueMerge(profile.avoid, updates.avoid),
    };
};

export const Chatbot: React.FC<Props> = ({ userId }) => {
    const [threads, setThreads] = useState<ChatThread[]>(() => loadThreads(userId));
    const [activeThreadId, setActiveThreadId] = useState(() => getInitialActiveThreadId(userId, loadThreads(userId)));
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [skinProfile, setSkinProfile] = useState<SkinProfile>(() => loadSkinProfile(userId));
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const activeThread = useMemo(
        () => threads.find(thread => thread.id === activeThreadId) || threads[0],
        [activeThreadId, threads]
    );
    const messages = useMemo(
        () => activeThread?.messages || [createWelcomeMessage()],
        [activeThread]
    );

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        const loadedThreads = loadThreads(userId);
        setThreads(loadedThreads);
        setActiveThreadId(getInitialActiveThreadId(userId, loadedThreads));
        setSkinProfile(loadSkinProfile(userId));
        setInput('');
        setLoading(false);
    }, [userId]);

    useEffect(() => {
        try {
            localStorage.setItem(getThreadsStorageKey(userId), JSON.stringify(serializeThreads(threads)));
            localStorage.setItem(getActiveThreadStorageKey(userId), activeThreadId);
            localStorage.removeItem(getLegacyStorageKey(userId));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }, [activeThreadId, threads, userId]);

    useEffect(() => {
        try {
            localStorage.setItem(getSkinProfileStorageKey(userId), JSON.stringify(skinProfile));
        } catch (error) {
            console.error('Failed to save skin profile:', error);
        }
    }, [skinProfile, userId]);

    const updateActiveMessages = (updater: (current: Message[]) => Message[]) => {
        setThreads(prevThreads => prevThreads.map(thread => {
            if (thread.id !== activeThreadId) return thread;
            const nextMessages = updater(thread.messages);
            return {
                ...thread,
                title: deriveThreadTitle(nextMessages),
                updatedAt: new Date(),
                messages: nextMessages,
            };
        }));
    };

    const handleNewChat = () => {
        const nextThread = createEmptyThread();
        setThreads(prevThreads => [nextThread, ...prevThreads]);
        setActiveThreadId(nextThread.id);
        setInput('');
        setLoading(false);
    };

    const handleSelectThread = (threadId: string) => {
        setActiveThreadId(threadId);
        setInput('');
        setLoading(false);
    };

    const handleDeleteThread = (threadId: string) => {
        setThreads(prevThreads => {
            const remainingThreads = prevThreads.filter(thread => thread.id !== threadId);
            if (remainingThreads.length === 0) {
                const replacement = createEmptyThread();
                setActiveThreadId(replacement.id);
                return [replacement];
            }
            if (threadId === activeThreadId) {
                setActiveThreadId(remainingThreads[0].id);
            }
            return remainingThreads;
        });
    };

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        const userMessage: Message = {
            id: Date.now().toString(),
            type: 'user',
            content: input,
            timestamp: new Date(),
        };

        const conversationHistory = messages
            .filter(m => m.id !== 'welcome' && m.content.trim())
            .slice(-12)
            .map(m => ({
                role: m.type === 'user' ? 'user' : 'assistant',
                content: m.content,
            }));

        updateActiveMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            // Call the backend API
            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            
            // 🔍 DEBUG LOGGING
            console.group('🤖 Chatbot API Request');
            console.log('API_URL:', API_URL);
            console.log('Full URL:', `${API_URL}/api/chat`);
            console.log('Origin:', window.location.origin);
            console.log('User Input:', input);
            console.groupEnd();
            
            const response = await fetch(`${API_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    message: input,
                    conversation_history: conversationHistory,
                    skin_profile: skinProfile,
                }),
            });

            console.log('📡 Response Status:', response.status);
            console.log('📡 Response Headers:', Object.fromEntries(response.headers.entries()));

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Response Data:', data);

            if (data.memory_updates) {
                setSkinProfile(prevProfile => mergeSkinProfile(prevProfile, data.memory_updates));
            }
            
            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: data.response || 'I couldn\'t process that. Please try again.',
                timestamp: new Date(),
                products: data.products || []
            };

            updateActiveMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage: Message = {
                id: (Date.now() + 2).toString(),
                type: 'bot',
                content: 'Sorry, I encountered an error. Please check your connection and try again.',
                timestamp: new Date(),
            };
            updateActiveMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-50 max-w-7xl mx-auto border-x border-gray-200">
            <aside className="hidden md:flex w-72 shrink-0 flex-col border-r border-gray-200 bg-white">
                <div className="p-4 border-b border-gray-200">
                    <button
                        type="button"
                        onClick={handleNewChat}
                        disabled={loading}
                        className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
                    >
                        New chat
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                    <p className="px-2 text-xs font-bold uppercase tracking-wide text-gray-400">Chat history</p>
                    {threads
                        .slice()
                        .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
                        .map(thread => (
                            <div
                                key={thread.id}
                                className={`group flex items-center gap-2 rounded-lg border px-3 py-2 ${
                                    thread.id === activeThreadId
                                        ? 'border-blue-200 bg-blue-50'
                                        : 'border-transparent hover:bg-gray-50'
                                }`}
                            >
                                <button
                                    type="button"
                                    onClick={() => handleSelectThread(thread.id)}
                                    className="min-w-0 flex-1 text-left"
                                >
                                    <span className="block truncate text-sm font-medium text-gray-900">{thread.title}</span>
                                    <span className="block text-xs text-gray-500">
                                        {thread.updatedAt.toLocaleDateString([], { month: 'short', day: 'numeric' })}
                                    </span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleDeleteThread(thread.id)}
                                    className="rounded px-2 py-1 text-xs text-gray-400 opacity-0 hover:bg-gray-100 hover:text-red-600 group-hover:opacity-100"
                                    aria-label={`Delete ${thread.title}`}
                                >
                                    Delete
                                </button>
                            </div>
                        ))}
                </div>
            </aside>

            <div className="flex min-w-0 flex-1 flex-col">
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 shadow-lg flex items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-bold">🤖 AI Skin Health Assistant</h1>
                        <p className="text-blue-100 text-sm mt-1">Ask any questions about skin care and conditions</p>
                    </div>
                    <button
                        type="button"
                        onClick={handleNewChat}
                        disabled={loading}
                        className="md:hidden shrink-0 rounded-lg border border-white/30 bg-white/10 px-3 py-2 text-sm font-medium text-white hover:bg-white/20 disabled:opacity-50"
                    >
                        New chat
                    </button>
                </div>

                {/* Messages Container */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`flex items-end ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            {message.type === 'bot' && (
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2 flex-shrink-0 border border-blue-200">
                                    <span role="img" aria-label="robot" className="text-sm">🤖</span>
                                </div>
                            )}

                            <div
                                className={`max-w-[80%] md:max-w-[70%] px-4 py-3 rounded-2xl shadow-sm ${
                                    message.type === 'user'
                                        ? 'bg-blue-600 text-white rounded-br-none'
                                        : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                                }`}
                            >
                                <div className={`text-sm leading-relaxed max-w-none ${
                                    message.type === 'user'
                                        ? '[&_a]:text-white [&_ul]:list-disc [&_ol]:list-decimal [&_li]:ml-4 [&_p]:my-1 [&_strong]:font-semibold'
                                        : 'text-gray-800 [&_a]:text-blue-600 [&_ul]:list-disc [&_ol]:list-decimal [&_li]:ml-4 [&_p]:my-1 [&_strong]:font-semibold [&_h1]:text-base [&_h2]:text-base [&_h3]:text-sm [&_h1]:font-bold [&_h2]:font-bold [&_h3]:font-bold'
                                }`}>
                                    <ReactMarkdown>{message.content}</ReactMarkdown>
                                </div>
                                
                                {/* Render Product Cards if available */}
                                {message.products && message.products.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-gray-100">
                                        <p className="text-xs font-bold text-gray-500 uppercase mb-2">Recommended Products</p>
                                        <div className="flex gap-3 overflow-x-auto pb-2 snap-x">
                                            {message.products.map((product, idx) => (
                                                <div key={idx} className="min-w-[200px] max-w-[200px] snap-center">
                                                    <ProductCard product={product} compact={true} />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <span className={`text-xs mt-1 block ${
                                    message.type === 'user' ? 'text-blue-100' : 'text-gray-400'
                                }`}>
                                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>

                            {message.type === 'user' && (
                                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center ml-2 flex-shrink-0 border border-gray-300">
                                    <span role="img" aria-label="user" className="text-sm">👤</span>
                                </div>
                            )}
                        </div>
                    ))}
                    
                    {loading && (
                        <div className="flex items-end justify-start">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2 flex-shrink-0 border border-blue-200">
                                <span role="img" aria-label="robot" className="text-sm">🤖</span>
                            </div>
                            <div className="bg-white text-gray-800 border border-gray-200 px-4 py-3 rounded-2xl rounded-bl-none shadow-sm">
                                <div className="flex space-x-2">
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                                </div>
                            </div>
                        </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="border-t border-gray-200 bg-white p-4">
                    <form onSubmit={handleSendMessage} className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask me about skin conditions, treatments, or skincare tips..."
                            disabled={loading}
                            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                        />
                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                        >
                            Send
                        </button>
                    </form>
                    <p className="text-xs text-gray-500 mt-2">
                        💡 Tip: Ask about symptoms, treatments, active ingredients, product recommendations, or skin conditions.
                    </p>
                </div>
            </div>
        </div>
    );
};
