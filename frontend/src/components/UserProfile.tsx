import React, { useState, useEffect } from 'react';
import { fetchUserAttributes, updateUserAttribute } from 'aws-amplify/auth';

interface Props {
    user: any;
    onSignOut: () => void;
}

export const UserProfile: React.FC<Props> = ({ user, onSignOut }) => {
    const [attributes, setAttributes] = useState<any>({});
    const [isEditingName, setIsEditingName] = useState(false);
    const [firstName, setFirstName] = useState('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (user?.given_name) {
            setAttributes(user);
            setFirstName(user.given_name);
            setLoading(false);
        } else {
            loadAttributes();
        }
    }, [user]);

    const loadAttributes = async () => {
        try {
            const attrs = await fetchUserAttributes();
            setAttributes(attrs);
            setFirstName(attrs.given_name || '');
        } catch (error) {
            console.error('Error fetching attributes:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdateName = async () => {
        try {
            await updateUserAttribute({
                userAttribute: {
                    attributeKey: 'given_name',
                    value: firstName
                }
            });
            setAttributes({ ...attributes, given_name: firstName });
            setIsEditingName(false);
        } catch (error) {
            console.error('Error updating name:', error);
            alert('Failed to update name');
        }
    };

    return (
        <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-8 text-white text-center">
                <div className="w-24 h-24 bg-white rounded-full mx-auto flex items-center justify-center text-blue-600 text-4xl font-bold shadow-lg mb-4">
                    {loading ? (
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    ) : (
                        attributes.given_name ? attributes.given_name[0].toUpperCase() : (user?.username?.[0]?.toUpperCase() || 'U')
                    )}
                </div>
                
                {isEditingName ? (
                    <div className="flex items-center justify-center gap-2 mb-2">
                        <input
                            type="text"
                            value={firstName}
                            onChange={(e) => setFirstName(e.target.value)}
                            className="px-3 py-1 text-gray-800 rounded-lg border-none focus:ring-2 focus:ring-blue-300 outline-none w-40 text-center font-bold"
                            placeholder="First Name"
                            autoFocus
                        />
                        <button 
                            onClick={handleUpdateName}
                            className="bg-white/20 hover:bg-white/30 p-1 rounded-full transition-colors"
                            title="Save"
                        >
                            ✅
                        </button>
                        <button 
                            onClick={() => setIsEditingName(false)}
                            className="bg-white/20 hover:bg-white/30 p-1 rounded-full transition-colors"
                            title="Cancel"
                        >
                            ❌
                        </button>
                    </div>
                ) : (
                    <h2 className="text-2xl font-bold flex items-center justify-center gap-2 group min-h-[2rem]">
                        {loading ? (
                            <span className="opacity-50 text-sm">Loading profile...</span>
                        ) : (
                            <>
                                {attributes.given_name || 'User'}
                                <button 
                                    onClick={() => {
                                        setFirstName(attributes.given_name || '');
                                        setIsEditingName(true);
                                    }}
                                    className="opacity-0 group-hover:opacity-100 transition-opacity text-sm bg-white/20 hover:bg-white/30 p-1 rounded"
                                    title="Edit Name"
                                >
                                    ✏️
                                </button>
                            </>
                        )}
                    </h2>
                )}
                
                <p className="text-blue-100">{user?.signInDetails?.loginId || user?.email || attributes.email || 'No email'}</p>
            </div>

            <div className="p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Account ID</h3>
                        <p className="font-mono text-gray-800 truncate" title={user?.userId}>{user?.userId}</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Member Since</h3>
                        <p className="text-gray-800">{new Date().toLocaleDateString()}</p>
                    </div>
                </div>

                <div className="border-t border-gray-200 pt-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Skin Profile</h3>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg border border-blue-100">
                            <div>
                                <p className="font-medium text-blue-900">Skin Type</p>
                                <p className="text-sm text-blue-700">Not set</p>
                            </div>
                            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">Edit</button>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg border border-purple-100">
                            <div>
                                <p className="font-medium text-purple-900">Concerns</p>
                                <p className="text-sm text-purple-700">None selected</p>
                            </div>
                            <button className="text-purple-600 hover:text-purple-800 text-sm font-medium">Edit</button>
                        </div>
                    </div>
                </div>

                <div className="border-t border-gray-200 pt-6">
                    <button
                        onClick={onSignOut}
                        className="w-full bg-red-50 hover:bg-red-100 text-red-600 font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                    >
                        <span>🚪</span> Sign Out
                    </button>
                </div>
            </div>
        </div>
    );
};
