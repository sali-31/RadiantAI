export interface Product {
    // Product interface for recommendations
    name?: string; // Some components might use name
    title?: string; // Backend returns title
    category?: string;
    id?: string;
    brand?: string;
    asin?: string;
    price?: number | string;
    price_numeric?: number;
    rating?: number;
    reviews?: number;
    ingredients?: string;
    combination?: number;
    dry?: number;
    normal?: number;
    oily?: number;
    sensitive?: number;
    image_url?: string;
    product_url?: string;
    thumbnail?: string;
    link?: string;
    reason?: string;
    directions?: string;
    value_score?: number;
    condition?: string;
    data_source?: string;
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
        bundle?: Product[];
        recommendations?: Product[];
        full_catalog?: Product[];
        total_cost?: number;
        budget_max?: number;
    };
}

export interface AppUser {
    userId?: string;
    username?: string;
    given_name?: string;
    email?: string;
    signInDetails?: {
        loginId?: string;
    };
}

export const getErrorMessage = (error: unknown, fallback: string) => {
    return error instanceof Error ? error.message : fallback;
};
