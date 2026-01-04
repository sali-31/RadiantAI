import React, { useState } from 'react';
import type { AnalysisResponse } from '../types';
import { ProductRoutine } from './ProductRoutine';

interface Props {
    data: AnalysisResponse;
    onBack: () => void;
}

export const RecommendedProducts = ({ data, onBack }: Props) => {
    const [budget, setBudget] = useState<string>('');
    const [currentBundle, setCurrentBundle] = useState(data.product_recommendations.bundle || []);
    const [allRecommendations, setAllRecommendations] = useState(data.product_recommendations.recommendations || []);
    const [fullCatalog, setFullCatalog] = useState(data.product_recommendations.full_catalog || []);
    const [sortBy, setSortBy] = useState<'default' | 'price_asc' | 'price_desc' | 'rating' | 'reviews'>('default');
    const [isUpdating, setIsUpdating] = useState(false);
    const [currentPage, setCurrentPage] = useState(1);
    const ITEMS_PER_PAGE = 8;
    const [bundleStats, setBundleStats] = useState({
        totalCost: data.product_recommendations.total_cost || 0,
        savings: 0
    });
    const [lastUpdatedBudget, setLastUpdatedBudget] = useState<string | null>(null);

    // Automatically fetch bundle if it's empty but we have analysis
    React.useEffect(() => {
        if (currentBundle.length === 0 && data.ai_analysis.analysis) {
            handleUpdateBundle();
        }
    }, []);

    const handleUpdateBundle = async () => {
        setIsUpdating(true);
        try {
            const payload = {
                analysis_text: data.ai_analysis.analysis,
                budget_max: budget ? parseFloat(budget) : null
            };

            const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await fetch(`${API_URL}/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const result = await response.json();
                setCurrentBundle(result.bundle);
                setAllRecommendations(result.recommendations || []);
                setFullCatalog(result.full_catalog || []);
                setBundleStats({
                    totalCost: result.total_cost,
                    savings: 0
                });
                setLastUpdatedBudget(budget || 'No Limit');
            } else {
                console.error("Failed to update bundle");
            }
        } catch (error) {
            console.error("Error updating bundle:", error);
        } finally {
            setIsUpdating(false);
        }
    };

    const getSortedProducts = (products: any[], sort: string) => {
        if (sort === 'default') return products;
        
        return [...products].sort((a, b) => {
            switch (sort) {
                case 'price_asc':
                    return (a.price_numeric || 0) - (b.price_numeric || 0);
                case 'price_desc':
                    return (b.price_numeric || 0) - (a.price_numeric || 0);
                case 'rating':
                    return (b.rating || 0) - (a.rating || 0);
                case 'reviews':
                    return (b.reviews || 0) - (a.reviews || 0);
                default:
                    return 0;
            }
        });
    };

    const getSectionTitle = () => {
        switch (sortBy) {
            case 'price_asc': return 'Best Value Picks';
            case 'price_desc': return 'Premium Picks';
            case 'rating': return 'Top Rated Picks';
            case 'reviews': return 'Most Popular Picks';
            default: return 'Top Picks';
        }
    };

    // Derived state for display
    const displayedCatalog = getSortedProducts(fullCatalog, sortBy);
    const displayedTopPicks = sortBy === 'default' 
        ? allRecommendations 
        : displayedCatalog.slice(0, 5); // When sorted, show top 5 from the sorted full catalog

    return (
        <div className="max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
            <button onClick={onBack} className="mb-6 text-blue-600 hover:underline flex items-center font-medium">
                <span className="mr-2">←</span> Back to Analysis
            </button>

            <div className="border-b pb-6 mb-8">
                <h2 className="text-3xl font-bold text-gray-900">Recommended Products</h2>
                <p className="text-gray-500 mt-2">
                    Based on your skin analysis, we've curated this routine for you. 
                    Adjust your budget to see different options.
                </p>
            </div>

            <div className="flex flex-col md:flex-row justify-between items-end mb-8">
                <div className="text-center md:text-left w-full md:w-auto mb-4 md:mb-0">
                    <h3 className="text-xl font-semibold text-gray-800">Your Personalized Bundle</h3>
                </div>
                
                {/* Budget Control */}
                <div className="flex flex-col items-end">
                    <div className="flex items-center bg-gray-50 p-3 rounded-lg border border-gray-200 shadow-sm">
                        <label className="text-sm font-medium text-gray-700 mr-3">Max Budget:</label>
                        <div className="relative">
                            <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                            <input 
                                type="number" 
                                value={budget}
                                onChange={(e) => setBudget(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleUpdateBundle()}
                                placeholder="No Limit"
                                className="pl-7 pr-3 py-1.5 w-28 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-right"
                                min="0"
                            />
                        </div>
                        <button 
                            onClick={handleUpdateBundle}
                            disabled={isUpdating}
                            className="ml-3 px-4 py-1.5 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
                        >
                            {isUpdating ? 'Updating...' : 'Update Bundle'}
                        </button>
                    </div>
                    {lastUpdatedBudget !== null && (
                        <span className="text-xs text-gray-500 mt-1 mr-1">
                            Active Budget: <span className="font-medium">{lastUpdatedBudget === 'No Limit' ? 'No Limit' : `$${lastUpdatedBudget}`}</span>
                        </span>
                    )}
                </div>
            </div>

            {/* Bundle Summary */}
            {currentBundle.length > 0 && (
                <div className="mb-6 flex justify-end">
                    <div className="text-right">
                        <span className="text-gray-600 text-sm mr-2">Total Bundle Cost:</span>
                        <span className="text-xl font-bold text-green-600">${bundleStats.totalCost.toFixed(2)}</span>
                    </div>
                </div>
            )}
            
            {currentBundle.length > 0 ? (
                <div className="max-w-3xl mx-auto mb-8">
                    <ProductRoutine products={currentBundle} />
                </div>
            ) : (
                <div className="bg-gray-50 p-6 rounded-xl text-center text-gray-500 mb-12">
                    No specific product recommendations available for this budget. Try increasing it.
                </div>
            )}

            {/* Individual Recommendations Section */}
            {displayedTopPicks.length > 0 && (
                <div className="border-t pt-10 mb-12">
                    <div className="flex flex-col sm:flex-row justify-between items-center mb-6 gap-4">
                        <h3 className="text-2xl font-bold text-gray-900">{getSectionTitle()}</h3>
                        <div className="flex items-center gap-3">
                            <label className="text-sm font-medium text-gray-700">Sort by:</label>
                            <select 
                                value={sortBy}
                                onChange={(e) => setSortBy(e.target.value as any)}
                                className="block w-48 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md border"
                            >
                                <option value="default">Default (Recommended)</option>
                                <option value="price_asc">Price: Low to High</option>
                                <option value="price_desc">Price: High to Low</option>
                                <option value="rating">Highest Rated</option>
                                <option value="reviews">Most Reviewed</option>
                            </select>
                        </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {displayedTopPicks.map((product: any, idx: number) => (
                            <ProductCard key={idx} product={product} />
                        ))}
                    </div>
                </div>
            )}

            {/* Full Catalog Section */}
            {displayedCatalog.length > 0 && (
                <div className="border-t pt-10">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6">All Matching Products</h3>
                    <p className="text-gray-500 mb-6">Complete catalog of products matching your skin analysis.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {displayedCatalog.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE).map((product: any, idx: number) => (
                            <ProductCard key={`catalog-${idx}`} product={product} />
                        ))}
                    </div>

                    {/* Pagination Controls */}
                    {displayedCatalog.length > ITEMS_PER_PAGE && (
                        <div className="flex justify-center items-center mt-8 gap-4">
                            <button
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                disabled={currentPage === 1}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Previous
                            </button>
                            <span className="text-sm text-gray-600 font-medium">
                                Page {currentPage} of {Math.ceil(displayedCatalog.length / ITEMS_PER_PAGE)}
                            </span>
                            <button
                                onClick={() => setCurrentPage(p => Math.min(Math.ceil(displayedCatalog.length / ITEMS_PER_PAGE), p + 1))}
                                disabled={currentPage >= Math.ceil(displayedCatalog.length / ITEMS_PER_PAGE)}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Next
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export const ProductCard = ({ product, compact = false }: { product: any, compact?: boolean }) => (
    <div className={`border rounded-lg hover:shadow-md transition-shadow bg-white flex flex-col h-full ${compact ? 'p-3 min-w-[180px]' : 'p-4'}`}>
        <div className={`${compact ? 'h-32' : 'h-48'} flex items-center justify-center mb-4 bg-gray-50 rounded-md overflow-hidden relative group`}>
            {product.thumbnail ? (
                <img src={product.thumbnail} alt={product.title} className="max-h-full max-w-full object-contain transition-transform group-hover:scale-105" />
            ) : (
                <div className="text-gray-400 text-xs">No Image</div>
            )}
        </div>
        <div className="grow">
            <div className="text-xs font-bold text-blue-600 uppercase mb-1">{product.category || 'Product'}</div>
            <h4 className={`font-medium text-gray-900 line-clamp-2 mb-2 ${compact ? 'text-xs' : 'text-sm'}`} title={product.title}>
                {product.title}
            </h4>
            <div className="flex items-center mb-2">
                <span className="text-yellow-400 mr-1 text-xs">★</span>
                <span className="text-xs text-gray-600">{product.rating} ({product.reviews})</span>
            </div>
        </div>
        <div className={`flex items-center justify-between border-t border-gray-100 ${compact ? 'mt-2 pt-2' : 'mt-4 pt-4'}`}>
            <span className={`${compact ? 'text-sm' : 'text-lg'} font-bold text-gray-900`}>
                ${typeof product.price_numeric === 'number' ? product.price_numeric.toFixed(2) : product.price}
            </span>
            <a 
                href={product.link} 
                target="_blank" 
                rel="noopener noreferrer"
                className={`bg-gray-900 text-white font-medium rounded hover:bg-gray-800 transition-colors ${compact ? 'px-2 py-1 text-[10px]' : 'px-3 py-1.5 text-xs'}`}
            >
                View
            </a>
        </div>
    </div>
);
