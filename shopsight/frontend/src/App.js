import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import ProductCard from './components/ProductCard';
import ProductInsights from './components/ProductInsights';
import Dashboard from './components/Dashboard';
import Hero from './components/Hero';
import HighlightsSection from './components/HighlightsSection';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [searchResults, setSearchResults] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productInsights, setProductInsights] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('search');

  useEffect(() => {
    // Load dashboard stats on startup
    loadDashboardStats();
  }, []);

  const loadDashboardStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
      const data = await response.json();
      setDashboardStats(data);
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
    }
  };

  const handleSearch = async (query) => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, limit: 12 }),
      });
      
      const data = await response.json();
      setSearchResults(data.products || []);
      setActiveTab('search');
    } catch (error) {
      console.error('Error searching products:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleProductSelect = async (product) => {
    setSelectedProduct(product);
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/product/${product.article_id}/insights`);
      const insights = await response.json();
      setProductInsights(insights);
      setActiveTab('insights');
    } catch (error) {
      console.error('Error loading product insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const scrollToSearch = () => {
    const searchSection = document.getElementById('search-section');
    if (searchSection) {
      searchSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white/80 backdrop-blur border-b border-slate-100 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100">
                <span className="text-xl font-semibold text-blue-600">SS</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">ShopSight</h1>
                <span className="text-xs uppercase tracking-wide text-gray-400">Counterfactual commerce studio</span>
              </div>
            </div>
            <nav className="flex items-center space-x-6 text-sm font-medium text-gray-600">
              <button
                onClick={() => {
                  setActiveTab('search');
                  scrollToSearch();
                }}
                className="hover:text-gray-900"
              >
                Demo
              </button>
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-3 py-2 rounded-lg transition ${
                  activeTab === 'dashboard'
                    ? 'bg-blue-100 text-blue-700'
                    : 'hover:bg-gray-100 text-gray-600'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => window.open('https://github.com/your-org/shopsight', '_blank')}
                className="rounded-lg border border-gray-200 px-3 py-2 text-gray-600 transition hover:border-gray-300 hover:text-gray-900"
              >
                GitHub
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero */}
      <main className="max-w-7xl mx-auto space-y-16 py-12 sm:px-6 lg:px-8">
        <Hero stats={dashboardStats} onGetStarted={() => {
          setActiveTab('search');
          scrollToSearch();
        }} />

        <HighlightsSection onDemoClick={() => {
          setActiveTab('search');
          scrollToSearch();
        }} />

        <div id="search-section" className="space-y-10">
            {/* Search Section */}
            <div className="bg-white rounded-2xl shadow-lg border border-slate-100 p-6">
              <SearchBar onSearch={handleSearch} loading={loading} />
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg border border-slate-100">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-900">
                    Search Results ({searchResults.length} products)
                  </h2>
                </div>
                <div className="p-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {searchResults.map((product) => (
                      <ProductCard
                        key={product.article_id}
                        product={product}
                        onSelect={() => handleProductSelect(product)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Product Insights */}
            {selectedProduct && productInsights && (
              <ProductInsights
                product={selectedProduct}
                insights={productInsights}
                onClose={() => {
                  setSelectedProduct(null);
                  setProductInsights(null);
                }}
              />
            )}
          </div>
        )}

        {activeTab === 'dashboard' && (
          <Dashboard stats={dashboardStats} />
        )}
      </main>

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span className="text-gray-700">Loading...</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
