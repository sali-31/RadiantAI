import { useMemo, useState, useEffect } from 'react';
import type { AnalysisResponse } from '../types';

interface Props {
    data: AnalysisResponse;
    onBack: () => void;
    onViewProducts: () => void;
}

interface DiagnosisData {
    characterization: string;
    severity: string;
    location: string;
    recommendation: string;
    treatments: string[];
    blemish_regions: Array<{
        type: string;
        x_min: number;
        y_min: number;
        x_max: number;
        y_max: number;
        confidence: number;
        severity?: string;
    }>;
}

export const AnalysisResults = ({ data, onBack, onViewProducts }: Props) => {
    const [imageUrl, setImageUrl] = useState(data.s3_path);

    // Effect to refresh the URL if it might be expired
    useEffect(() => {
        const refreshUrl = async () => {
            // If we have a key, try to get a fresh URL
            try {
                const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
                if (data.s3_key) {
                    const response = await fetch(`${API_URL}/api/image-url`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ s3_key: data.s3_key }),
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        setImageUrl(result.url);
                    }
                }
            } catch (error) {
                console.error("Failed to refresh image URL:", error);
            }
        };

        refreshUrl();
    }, [data.s3_key]);

    // Parse the JSON string from the AI analysis
    const diagnosis = useMemo(() => {
        try {
            // The backend returns a JSON string inside the 'analysis' field
            return JSON.parse(data.ai_analysis.analysis) as DiagnosisData;
        } catch (e) {
            console.error("Failed to parse AI analysis JSON:", e);
            return null;
        }
    }, [data.ai_analysis.analysis]);

    if (!diagnosis) {
        return (
            <div className="p-8 text-center text-red-600">
                Error parsing analysis results. Please try again.
            </div>
        );
    }

    return (
        <div className="w-full min-h-screen bg-gray-50 p-4 md:p-8">
            <div className="max-w-[1600px] mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
                {/* Header Bar */}
                <div className="bg-white border-b border-gray-100 p-6 flex items-center justify-between sticky top-0 z-10">
                    <button onClick={onBack} className="text-gray-600 hover:text-blue-600 flex items-center font-medium transition-colors">
                        <span className="mr-2 text-xl">←</span> Back to Dashboard
                    </button>
                </div>

                <div className="flex flex-col xl:flex-row">
                    {/* Left Column: Image (Sticky on Desktop) */}
                    <div className="w-full xl:w-1/2 bg-gray-50 p-6 md:p-10 flex flex-col items-center justify-center border-b xl:border-b-0 xl:border-r border-gray-200">
                        <div className="relative w-full max-w-2xl aspect-4/3 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                            <img 
                                src={imageUrl} 
                                alt="Analyzed Skin" 
                                className="w-full h-full object-contain"
                                onError={(e) => {
                                    e.currentTarget.src = 'https://placehold.co/600x400?text=Image+Protected';
                                }}
                            />
                            {/* Overlay Bounding Boxes - DISABLED FOR NOW */}
                            {false && diagnosis?.blemish_regions?.map((region, idx) => (
                                <div
                                    key={idx}
                                    className="absolute border-2 border-red-500 bg-red-500/10 hover:bg-red-500/20 transition-colors cursor-help"
                                    style={{
                                        left: `${region.x_min * 100}%`,
                                        top: `${region.y_min * 100}%`,
                                        width: `${(region.x_max - region.x_min) * 100}%`,
                                        height: `${(region.y_max - region.y_min) * 100}%`,
                                    }}
                                    title={`${region.type}${region.severity ? ` - ${region.severity}` : ''} (${Math.round(region.confidence * 100)}% confidence)`}
                                />
                            ))}
                        </div>
                        <div className="mt-6 flex items-center text-sm text-gray-500 bg-white px-4 py-2 rounded-full shadow-sm border border-gray-200">
                            <span className="mr-2">🛡️</span>
                            Metadata scrubbed for privacy
                        </div>
                    </div>

                    {/* Right Column: Analysis Details (Scrollable) */}
                    <div className="w-full xl:w-1/2 p-6 md:p-10 space-y-8 bg-white">
                        <div>
                            <h2 className="text-3xl font-bold text-gray-900 mb-2">Skin Analysis Results</h2>
                            <p className="text-gray-500">AI-Powered Dermatological Assessment</p>
                        </div>

                        <div className="space-y-6">
                            {/* Observation Card */}
                            <div className="bg-blue-50/50 rounded-xl p-6 border border-blue-100">
                                <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
                                    <span className="w-1.5 h-6 bg-blue-500 rounded-full mr-3"></span>
                                    Observation
                                </h3>
                                <p className="text-gray-700 leading-relaxed text-lg">
                                    {diagnosis.characterization}
                                </p>
                            </div>

                            {/* Location & Treatments Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="bg-purple-50/50 rounded-xl p-6 border border-purple-100">
                                    <h3 className="text-lg font-semibold text-purple-900 mb-3 flex items-center">
                                        <span className="w-1.5 h-6 bg-purple-500 rounded-full mr-3"></span>
                                        Affected Areas
                                    </h3>
                                    <p className="text-gray-700">{diagnosis.location}</p>
                                </div>

                                <div className="bg-teal-50/50 rounded-xl p-6 border border-teal-100">
                                    <h3 className="text-lg font-semibold text-teal-900 mb-3 flex items-center">
                                        <span className="w-1.5 h-6 bg-teal-500 rounded-full mr-3"></span>
                                        Active Ingredients
                                    </h3>
                                    <div className="flex flex-wrap gap-2">
                                        {diagnosis.treatments?.map((t, i) => (
                                            <span key={i} className="px-3 py-1 bg-white text-teal-700 rounded-lg text-sm font-medium border border-teal-100 shadow-sm">
                                                {t}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Recommendation Card */}
                            <div className="bg-amber-50 rounded-xl p-6 border border-amber-100">
                                <h3 className="text-lg font-semibold text-amber-900 mb-3 flex items-center">
                                    <span className="text-2xl mr-2">💡</span> Recommendation
                                </h3>
                                <p className="text-amber-900/80 leading-relaxed">
                                    {diagnosis.recommendation}
                                </p>
                            </div>
                        </div>

                        {/* CTA Section */}
                        <div className="pt-8 mt-8 border-t border-gray-100">
                            <div className="flex flex-col sm:flex-row items-center justify-between gap-6 bg-gray-900 text-white p-6 rounded-2xl shadow-lg">
                                <div>
                                    <h3 className="text-xl font-bold mb-1">Your Personalized Routine</h3>
                                    <p className="text-gray-400 text-sm">Based on this analysis</p>
                                </div>
                                <button
                                    onClick={onViewProducts}
                                    className="whitespace-nowrap px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/25"
                                >
                                    View Products →
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
