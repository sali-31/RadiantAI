import React, { useState, useRef, useEffect } from 'react';
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

export const Chatbot: React.FC<Props> = ({ userId }) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            type: 'bot',
            content: 'Hello! 👋 I\'m your AI skin health assistant. Ask me anything about skin conditions, treatments, skincare routines, or general dermatology advice. What would you like to know?',
            timestamp: new Date(),
        },
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

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

        setMessages(prev => [...prev, userMessage]);
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
                    conversation_history: messages.map(m => ({
                        role: m.type === 'user' ? 'user' : 'assistant',
                        content: m.content,
                    })),
                }),
            });

            console.log('📡 Response Status:', response.status);
            console.log('📡 Response Headers:', Object.fromEntries(response.headers.entries()));

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Response Data:', data);
            
            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: data.response || 'I couldn\'t process that. Please try again.',
                timestamp: new Date(),
                products: data.products || []
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage: Message = {
                id: (Date.now() + 2).toString(),
                type: 'bot',
                content: 'Sorry, I encountered an error. Please check your connection and try again.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-50 max-w-4xl mx-auto">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 shadow-lg">
                <h1 className="text-2xl font-bold">🤖 AI Skin Health Assistant</h1>
                <p className="text-blue-100 text-sm mt-1">Ask any questions about skin care and conditions</p>
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
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                            
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
    );
};
