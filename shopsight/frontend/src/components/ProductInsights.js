import React, { useState, useEffect } from 'react';
import { X, TrendingUp, Users, Target, Lightbulb, BarChart3, BadgeCheck } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import CounterfactualPanel from './CounterfactualPanel';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ProductInsights = ({ product, insights, onClose }) => {
  const [salesData, setSalesData] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const fetchSalesData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/product/${product.article_id}/sales-history`);
        const data = await response.json();
        if (isMounted) {
          setSalesData(data.chart_data);
        }
      } catch (error) {
        console.error('Error loading sales data:', error);
      }
    };

    fetchSalesData();

    return () => {
      isMounted = false;
    };
  }, [product.article_id]);

  const salesChartData = salesData ? {
    labels: salesData.labels,
    datasets: salesData.datasets.map((dataset, index) => ({
      ...dataset,
      borderColor: index === 0 ? '#3B82F6' : '#10B981',
      backgroundColor: index === 0 ? '#3B82F680' : '#10B98180',
      tension: 0.4,
    }))
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Sales History',
      },
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Revenue ($)',
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Units Sold',
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex flex-col gap-6 p-6 border-b border-gray-200 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="h-16 w-16 overflow-hidden rounded-2xl bg-slate-100">
              {product.image_url ? (
                <img src={product.image_url} alt={product.product_name} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-slate-400">👜</div>
              )}
            </div>
            <div>
              <div className="inline-flex items-center gap-2 rounded-full bg-blue-100 px-3 py-1 text-xs font-semibold text-blue-700">
                <BadgeCheck className="h-4 w-4" />
                AI Insight Ready
              </div>
              <h2 className="mt-2 text-2xl font-bold text-gray-900">{product.product_name}</h2>
              <p className="text-gray-600">{product.product_type} • {product.department}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="self-start rounded-full border border-gray-200 p-2 text-gray-400 transition hover:border-gray-300 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1.6fr_1fr] gap-6">
            <div className="space-y-6">
              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <TrendingUp className="h-8 w-8 text-blue-600" />
                    <div className="ml-3">
                      <p className="text-sm font-medium text-blue-600">Sales Trend</p>
                      <p className="text-lg font-semibold text-gray-900">{insights.sales_trend}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <Target className="h-8 w-8 text-green-600" />
                    <div className="ml-3">
                      <p className="text-sm font-medium text-green-600">Next Month Forecast</p>
                      <p className="text-lg font-semibold text-gray-900">
                        ${insights.forecast.next_month.toFixed(2)}
                      </p>
                      <p className="text-xs text-gray-500">
                        Confidence: {insights.forecast.confidence}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <Users className="h-8 w-8 text-purple-600" />
                    <div className="ml-3">
                      <p className="text-sm font-medium text-purple-600">Customer Segments</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {insights.customer_segments.length} Segments
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sales Chart */}
              {salesChartData && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <div className="flex items-center mb-4">
                    <BarChart3 className="h-5 w-5 text-gray-600 mr-2" />
                    <h3 className="text-lg font-medium text-gray-900">Sales Performance</h3>
                  </div>
                  <div className="h-64">
                    <Line data={salesChartData} options={chartOptions} />
                  </div>
                </div>
              )}

              {/* Customer Segments */}
              {insights.customer_segments.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Customer Segments</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {insights.customer_segments.map((segment, index) => (
                      <div key={index} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{segment.segment}</h4>
                          <span className="text-sm font-semibold text-blue-600">
                            {segment.percentage}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{segment.characteristics}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* AI Insights */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
                <div className="flex items-start">
                  <Lightbulb className="h-6 w-6 text-blue-600 mt-1 mr-3 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">AI-Generated Insights</h3>
                    <p className="text-gray-700 leading-relaxed whitespace-pre-line">{insights.insights}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <CounterfactualPanel />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductInsights;
