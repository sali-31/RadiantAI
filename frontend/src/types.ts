export interface Product {
    // Product interface for recommendations
    name?: string; // Some components might use name
    title?: string; // Backend returns title
    category?: string;
    id?: string;
    brand: string;
    price: number;
    rating: number;
    reviews: number;
    ingredients: string;
    combination: number;
    dry: number;
    normal: number;
    oily: number;
    sensitive: number;
    image_url?: string;
    product_url?: string;
    thumbnail?: string;
    link?: string;
    reason?: string;
}

export interface AnalysisResponse {
    message: string;
    s3_path: string;
    s3_key?: string; // Added for URL refreshing
    ai_analysis: {
        analysis: string;
        // Add other fields as your backend expands
    };
    product_recommendations: {
        bundle?: any[];
        recommendations?: any[];
        full_catalog?: any[];
        total_cost?: number;
        budget_max?: number;
    };
}
