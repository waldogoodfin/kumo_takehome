import React from 'react';
import { Sparkles, LineChart, Users } from 'lucide-react';

const Hero = ({ stats, onGetStarted }) => {
  const metrics = [
    {
      label: 'Products Indexed',
      value: stats ? stats.total_products.toLocaleString() : '105K+',
      icon: <Sparkles className="h-5 w-5 text-blue-500" />,
    },
    {
      label: 'Customer Profiles',
      value: stats ? stats.total_customers.toLocaleString() : '1.3M',
      icon: <Users className="h-5 w-5 text-indigo-500" />,
    },
    {
      label: 'Insights Generated',
      value: stats ? stats.total_transactions.toLocaleString() : '450K',
      icon: <LineChart className="h-5 w-5 text-purple-500" />,
    },
  ];

  return (
    <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 text-white shadow-xl">
      <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_top_left,#ffffff,transparent_45%)]" />
      <div className="relative px-6 py-16 sm:px-12 lg:px-16">
        <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center rounded-full bg-white/10 px-4 py-1 text-sm font-medium tracking-wide">
              <Sparkles className="mr-2 h-4 w-4" />
              AI-Powered E-commerce Intelligence
            </div>
            <div className="space-y-4">
              <h1 className="text-4xl font-bold leading-tight sm:text-5xl lg:text-6xl">
                Turn Catalog Data into Conversations and Counterfactual Insights
              </h1>
              <p className="max-w-xl text-lg text-white/80">
                ShopSight lets your team ask natural-language questions, surface real sales history, and visualize what happens if you launch the next campaign differently—all in one human-friendly workspace.
              </p>
            </div>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
              <button
                onClick={onGetStarted}
                className="inline-flex items-center justify-center rounded-xl bg-white px-6 py-3 text-base font-semibold text-blue-700 shadow-lg shadow-blue-900/20 transition hover:-translate-y-0.5 hover:bg-blue-50"
              >
                Explore the Demo
              </button>
              <p className="text-sm text-white/70">
                Try voice-like search and see revenue forecasts in seconds.
              </p>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -top-8 -left-8 h-32 w-32 rounded-full bg-white/10 blur-3xl" />
            <div className="relative rounded-2xl border border-white/10 bg-white/10 p-6 shadow-2xl backdrop-blur">
              <div className="space-y-6">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-wide text-white/60">Live insight preview</p>
                  <h3 className="text-2xl font-semibold">Copenhagen Strap Top</h3>
                  <p className="text-sm text-white/80">Forecasted to grow +18% next month with targeted influencer campaigns.</p>
                </div>
                <div className="grid gap-4">
                  <div className="rounded-xl bg-white/10 p-4">
                    <div className="flex items-baseline justify-between">
                      <span className="text-sm text-white/60">6-month revenue</span>
                      <span className="rounded-full bg-green-400/20 px-3 py-1 text-xs font-semibold text-green-100">+12.4%</span>
                    </div>
                    <p className="mt-2 text-xl font-semibold">$184,920</p>
                  </div>
                  <div className="rounded-xl bg-white/10 p-4">
                    <span className="text-sm text-white/60">Top customer cohort</span>
                    <p className="mt-2 font-medium">Trend Followers · 27%</p>
                    <p className="text-xs text-white/60">Respond to social-first drops within 48 hours.</p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  {metrics.map((metric) => (
                    <div key={metric.label} className="rounded-lg bg-black/10 p-3 text-center">
                      <div className="mx-auto mb-2 flex h-8 w-8 items-center justify-center rounded-full bg-white/10">
                        {metric.icon}
                      </div>
                      <p className="text-sm font-semibold">{metric.value}</p>
                      <p className="text-[11px] text-white/60">{metric.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
