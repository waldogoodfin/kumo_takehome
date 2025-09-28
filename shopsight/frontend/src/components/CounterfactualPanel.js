import React from 'react';
import { ArrowRightLeft, Rocket, Coins, Percent } from 'lucide-react';

const scenarios = [
  {
    name: 'Baseline',
    description: 'Current trajectory without new campaigns.',
    revenue: '$184,920',
    lift: '+0%',
    icon: <Percent className="h-5 w-5 text-slate-500" />,
    tone: 'text-slate-600',
    bg: 'bg-slate-50',
  },
  {
    name: 'Influencer Boost',
    description: 'Scale creator budget +20% targeting Trend Followers.',
    revenue: '$218,400',
    lift: '+18.1%',
    icon: <Rocket className="h-5 w-5 text-purple-500" />,
    tone: 'text-purple-600',
    bg: 'bg-purple-50',
  },
  {
    name: 'Bundle Offer',
    description: 'Pair with matching leggings; 15% bundle discount.',
    revenue: '$205,700',
    lift: '+11.3%',
    icon: <Coins className="h-5 w-5 text-amber-500" />,
    tone: 'text-amber-600',
    bg: 'bg-amber-50',
  },
];

const CounterfactualPanel = () => {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center gap-3 border-b border-slate-100 pb-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 text-blue-600">
          <ArrowRightLeft className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-900">Counterfactual Sandbox</h3>
          <p className="text-sm text-slate-500">
            Compare scenarios and share the lift forecast with stakeholders before launching.
          </p>
        </div>
      </div>

      <div className="mt-6 grid gap-4">
        {scenarios.map((scenario) => (
          <div
            key={scenario.name}
            className={`rounded-xl border border-transparent p-4 transition hover:border-slate-200 hover:shadow`}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${scenario.bg}`}>
                    {scenario.icon}
                  </div>
                  <h4 className="text-base font-semibold text-slate-900">{scenario.name}</h4>
                </div>
                <p className="mt-1 text-sm text-slate-500">{scenario.description}</p>
              </div>
              <div className="text-right">
                <p className="text-sm text-slate-500">Projected revenue</p>
                <p className="text-lg font-semibold text-slate-900">{scenario.revenue}</p>
                <p className={`text-sm font-medium ${scenario.tone}`}>{scenario.lift} vs baseline</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CounterfactualPanel;
