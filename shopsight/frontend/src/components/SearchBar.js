import React, { useState } from 'react';
import { Search, Sparkles } from 'lucide-react';

const SearchBar = ({ onSearch, loading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  const exampleQueries = [
    "Nike running shoes",
    "Summer dresses",
    "Blue jeans",
    "Winter coats",
    "Casual shirts"
  ];

  return (
    <div className="relative overflow-hidden rounded-3xl border border-slate-100 bg-white shadow-lg">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/40 via-transparent to-purple-50/30" />
      <div className="relative p-8">
        <div className="text-center space-y-3">
          <h2 className="text-3xl font-bold text-gray-900">
            Ask anything about your catalog or future campaigns
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto text-sm">
            ShopSight translates natural language into product matches, sales history, and counterfactual narratives so stakeholders get decisions, not dashboards.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-8 space-y-6">
          <div className="group relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-blue-500" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Try: ‘What if we promote breathable running jackets in Berlin next month?’"
              className="block w-full rounded-2xl border border-slate-200 bg-white/90 py-4 pl-12 pr-14 text-base shadow-sm transition focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              disabled={loading}
            />
            <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
              <Sparkles className="h-5 w-5 text-purple-500" />
            </div>
          </div>

          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => {
                    setQuery(example);
                    onSearch(example);
                  }}
                  className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-600 transition hover:border-blue-200 hover:bg-blue-100"
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="inline-flex items-center justify-center rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/30 transition hover:-translate-y-0.5 hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? 'Searching...' : 'Generate Insights'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SearchBar;
