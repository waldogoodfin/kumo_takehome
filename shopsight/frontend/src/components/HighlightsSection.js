import React from 'react';
import { Search, BarChart3, GitBranchPlus, Sparkle } from 'lucide-react';

const highlights = [
  {
    title: 'Conversational Product Discovery',
    description:
      'Let stakeholders type intent (“Nike running shoes under $120”) and watch the catalog respond with curated matches and instant context.',
    icon: <Search className="h-8 w-8 text-blue-500" />,
    accent: 'from-blue-500/10 to-blue-500/0',
  },
  {
    title: 'Sales & Forecasting at a Glance',
    description:
      'Unify historical performance with next-month forecasts and surface the drivers that matter—seasonality, campaigns, and customer cohorts.',
    icon: <BarChart3 className="h-8 w-8 text-indigo-500" />,
    accent: 'from-indigo-500/10 to-indigo-500/0',
  },
  {
    title: 'Counterfactual Playbooks',
    description:
      'Compare “what-if” scenarios like boosting influencer spend or bundling products, then visualize the projected lift before you commit.',
    icon: <GitBranchPlus className="h-8 w-8 text-purple-500" />,
    accent: 'from-purple-500/10 to-purple-500/0',
  },
  {
    title: 'Narratives, Not Dashboards',
    description:
      'LLM-generated takeaways translate dense charts into concise stories you can forward to merchandising, marketing, or finance teams.',
    icon: <Sparkle className="h-8 w-8 text-amber-500" />,
    accent: 'from-amber-500/10 to-amber-500/0',
  },
];

const HighlightsSection = ({ onDemoClick }) => {
  return (
    <section className="relative">
      <div className="mx-auto max-w-6xl space-y-10 px-6 sm:px-8">
        <div className="flex flex-col gap-4 text-center">
          <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">
            Your command center for retail “what if?” moments
          </h2>
          <p className="mx-auto max-w-3xl text-lg text-gray-600">
            ShopSight fuses product search, predictive analytics, and agentic storytelling so teams can move from questions to confident action in minutes.
          </p>
          <div className="flex justify-center">
            <button
              onClick={onDemoClick}
              className="inline-flex items-center rounded-full border border-blue-600 bg-white px-5 py-2 text-sm font-semibold text-blue-600 shadow-sm transition hover:bg-blue-50"
            >
              Jump to live demo
            </button>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {highlights.map((feature) => (
            <div
              key={feature.title}
              className="group relative overflow-hidden rounded-2xl border border-slate-100 bg-white p-6 shadow-sm transition hover:-translate-y-1 hover:shadow-xl"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${feature.accent} opacity-0 transition group-hover:opacity-100`} />
              <div className="relative flex items-start gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-900/5">
                  {feature.icon}
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold text-gray-900">{feature.title}</h3>
                  <p className="text-sm text-gray-600">{feature.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HighlightsSection;
