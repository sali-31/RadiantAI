
import type { Product } from '../types';

interface Props {
    products: Product[];
}

export const ProductRoutine = ({ products }: Props) => {
    // 1. Define the ideal order of steps
    const steps = ['Cleanser', 'Toner', 'Treatment', 'Moisturizer', 'Sunscreen'];

    // 2. Helper to guess the step based on product name/category
    const getStep = (product: Product) => {
        if (product.category) {
            // Capitalize first letter
            return product.category.charAt(0).toUpperCase() + product.category.slice(1);
        }
        const text = ((product.name || product.title || '') + (product.reason || '')).toLowerCase();
        if (text.includes('spf') || text.includes('sun')) return 'Sunscreen';
        if (text.includes('cleanse') || text.includes('wash')) return 'Cleanser';
        if (text.includes('moisturiz') || text.includes('cream')) return 'Moisturizer';
        if (text.includes('serum') || text.includes('acid')) return 'Treatment';
        if (text.includes('toner')) return 'Toner';
        return 'Essential'; // Fallback
    };

    // 3. Sort products into the routine order
    const sortedProducts = [...products].sort((a, b) => {
        const stepA = getStep(a);
        const stepB = getStep(b);
        
        // If both are "Essential" (unknown), keep original order
        if (stepA === 'Essential' && stepB === 'Essential') return 0;
        
        // If one is unknown, put it at the end
        if (stepA === 'Essential') return 1;
        if (stepB === 'Essential') return -1;

        // Check if steps are in our predefined list
        const indexA = steps.findIndex(s => stepA.includes(s));
        const indexB = steps.findIndex(s => stepB.includes(s));
        
        // If step not in list, put at end
        if (indexA === -1) return 1;
        if (indexB === -1) return -1;

        return indexA - indexB;
    });

    return (
        <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-4 shadow-sm">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {sortedProducts.map((product, index) => (
                    <div key={product.id || index} className="bg-white border border-blue-100 rounded-lg p-3 shadow-sm hover:shadow-md transition-all flex flex-col relative overflow-hidden">
                        {/* Step Badge */}
                        <div className="absolute top-0 left-0 bg-blue-600 text-white text-xs font-bold px-2 py-0.5 rounded-br-lg z-10">
                            Step {index + 1}: {getStep(product)}
                        </div>


                        <div className="h-32 flex items-center justify-center mb-3 mt-5 bg-gray-50 rounded-md overflow-hidden">
                            {product.thumbnail || product.image_url ? (
                                <img src={product.thumbnail || product.image_url} alt={product.name || product.title || 'Product'} className="max-h-full max-w-full object-contain" />
                            ) : (
                                <div className="text-gray-400 text-sm">No Image</div>
                            )}
                        </div>


                        <div className="flex-grow">
                            <h4 className="text-sm font-medium text-gray-900 line-clamp-2 mb-2" title={product.name || product.title}>
                                {product.name || product.title}
                            </h4>
                            {(product.rating !== undefined) && (
                                <div className="flex items-center mb-2">
                                    <span className="text-yellow-400 mr-1">★</span>
                                    <span className="text-sm text-gray-600">{product.rating} ({product.reviews || 0})</span>
                                </div>
                            )}
                        </div>

                        <div className="mt-4 flex items-center justify-between pt-3 border-t border-gray-100">
                            <span className="text-lg font-bold text-gray-900">
                                {typeof product.price === 'number' ? `$${product.price.toFixed(2)}` : product.price}
                            </span>
                            {(product.link || product.product_url) && (
                                <a 
                                    href={product.link || product.product_url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="px-3 py-1.5 bg-blue-600 text-white text-xs font-medium rounded hover:bg-blue-700 transition-colors"
                                >
                                    Buy Now
                                </a>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
