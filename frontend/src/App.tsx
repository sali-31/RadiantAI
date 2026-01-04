import { useState, useEffect } from 'react';
import { getCurrentUser, signOut, fetchUserAttributes } from 'aws-amplify/auth';
import { ImageUpload } from './components/ImageUpload';
import { AnalysisResults } from './components/AnalysisResults';
import { RecommendedProducts } from './components/RecommendedProducts';
import { Auth } from './components/Auth';
import { Dashboard } from './components/Dashboard';
import { UserProfile } from './components/UserProfile';
import { Chatbot } from './components/Chatbot';
import { ErrorBoundary } from './components/ErrorBoundary';
import type { AnalysisResponse } from './types';

type ViewState = 'dashboard' | 'upload' | 'results' | 'products' | 'chat' | 'analysis' | 'profile';
type TabState = 'dashboard' | 'analysis' | 'chat' | 'profile';

function App() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [currentView, setCurrentView] = useState<ViewState>('dashboard');
  const [activeTab, setActiveTab] = useState<TabState>('dashboard');
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    checkUser();
    
    // Load saved analysis from local storage
    const saved = localStorage.getItem('lesionrec_last_analysis');
    if (saved) {
      try {
        setAnalysisData(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse saved analysis");
        localStorage.removeItem('lesionrec_last_analysis');
      }
    }
  }, []);

  async function checkUser() {
    try {
      const currentUser = await getCurrentUser();
      const userAttributes = await fetchUserAttributes();
      setUser({ ...currentUser, ...userAttributes });
    } catch (err) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleSignOut() {
    try {
      await signOut();
      setUser(null);
      setAnalysisData(null);
      localStorage.removeItem('lesionrec_last_analysis');
      setCurrentView('dashboard');
    } catch (error) {
      console.error('Error signing out: ', error);
    }
  }

  const handleAnalysisComplete = (data: AnalysisResponse) => {
    setAnalysisData(data);
    localStorage.setItem('lesionrec_last_analysis', JSON.stringify(data));
    setCurrentView('results');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) {
    return <Auth onLoginSuccess={checkUser} />; 
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header with user info and sign out */}
      <div className="bg-white shadow-md border-b border-gray-200 relative z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 grid grid-cols-3 items-center">
          <div></div> {/* Spacer for centering */}
          
          <div className="flex justify-center">
            <h1 className="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-teal-500">
              Lumina
            </h1>
          </div>

          <div className="flex justify-end">
            <button
              onClick={() => {
                setActiveTab('profile');
                setCurrentView('profile');
              }}
              className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 text-white flex items-center justify-center text-lg font-bold hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg border-4 border-blue-50"
              title="User Profile"
            >
              {user?.given_name?.[0]?.toUpperCase() || user?.username?.[0]?.toUpperCase() || 'U'}
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="flex justify-center gap-4 flex-wrap">
              <button
                onClick={() => {
                  setActiveTab('dashboard');
                  setCurrentView('dashboard');
                }}
                className={`px-6 py-2.5 rounded-full font-medium transition-all duration-200 transform active:scale-95 ${
                  activeTab === 'dashboard'
                    ? 'bg-blue-600 text-white shadow-md ring-2 ring-blue-200'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900'
                }`}
              >
                📊 Dashboard
              </button>
              <button
                onClick={() => {
                  setActiveTab('analysis');
                  setCurrentView('upload');
                }}
                className={`px-6 py-2.5 rounded-full font-medium transition-all duration-200 transform active:scale-95 ${
                  activeTab === 'analysis'
                    ? 'bg-blue-600 text-white shadow-md ring-2 ring-blue-200'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900'
                }`}
              >
                🔍 Analysis
              </button>
              <button
                onClick={() => {
                  setActiveTab('chat');
                  setCurrentView('chat');
                }}
                className={`px-6 py-2.5 rounded-full font-medium transition-all duration-200 transform active:scale-95 ${
                  activeTab === 'chat'
                    ? 'bg-blue-600 text-white shadow-md ring-2 ring-blue-200'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900'
                }`}
              >
                🤖 AI Assistant
              </button>
            </div>
          </div>
        </div>

      {/* Main Content */}
      <div className="py-8">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && currentView === 'dashboard' && (
          <div className="max-w-6xl mx-auto px-6">
            <Dashboard 
              onStartAnalysis={() => {
                setActiveTab('analysis');
                setCurrentView('upload');
              }}
              onViewProducts={() => setCurrentView('products')}
              onViewResults={() => {
                setActiveTab('analysis');
                setCurrentView('results');
              }}
              analysisData={analysisData}
            />
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (currentView === 'upload' || currentView === 'results') && (
          <div className={currentView === 'results' ? "w-full" : "max-w-md mx-auto px-6"}>
            {currentView === 'upload' && (
              <>
                <div className="flex justify-between items-center mb-6">
                  <button 
                    onClick={() => {
                      setActiveTab('dashboard');
                      setCurrentView('dashboard');
                    }}
                    className="text-gray-500 hover:text-gray-700 flex items-center font-medium"
                  >
                    ← Dashboard
                  </button>

                  {analysisData && (
                    <button 
                      onClick={() => setCurrentView('results')}
                      className="bg-blue-50 text-blue-600 px-4 py-2 rounded-lg hover:bg-blue-100 transition-colors font-medium text-sm flex items-center gap-2"
                    >
                      📄 View Last Result
                    </button>
                  )}
                </div>
                <ImageUpload 
                  userId={user.userId} 
                  onAnalysisComplete={handleAnalysisComplete} 
                />
              </>
            )}
            {currentView === 'results' && analysisData && (
              <ErrorBoundary>
                <AnalysisResults 
                  data={analysisData} 
                  onBack={() => {
                    setActiveTab('dashboard');
                    setCurrentView('dashboard');
                  }} 
                  onViewProducts={() => setCurrentView('products')}
                />
              </ErrorBoundary>
            )}
          </div>
        )}

        {/* Products View */}
        {currentView === 'products' && analysisData && (
          <div className="max-w-6xl mx-auto px-6">
            <RecommendedProducts 
              data={analysisData} 
              onBack={() => setCurrentView('results')} 
            />
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && currentView === 'chat' && (
          <Chatbot userId={user.userId} />
        )}

        {/* Profile Tab */}
        {activeTab === 'profile' && currentView === 'profile' && (
          <UserProfile user={user} onSignOut={handleSignOut} />
        )}
      </div>
    </div>
  );
}

export default App;
