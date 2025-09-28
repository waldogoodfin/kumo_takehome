import React from 'react';
import { Eye, TrendingUp } from 'lucide-react';

const ProductCard = ({ product, onSelect }) => {
  return (
    <div
      className="product-card group relative flex flex-col overflow-hidden rounded-2xl border border-slate-100 bg-white shadow-sm transition hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"
      onClick={onSelect}
    >
      <div className="relative h-48 w-full overflow-hidden bg-slate-100">
        {product.image_url ? (
          <img
            src={product.image_url}
            alt={product.product_name}
            className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
            onError={(e) => {
              e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMDAgODBMMTIwIDEwMEgxMDBWODBaIiBmaWxsPSIjOUNBM0FGIi8+CjxwYXRoIGQ9Ik0xMDAgMTIwTDEyMCAxMDBIMTAwVjEyMFoiIGZpbGw9IiM5Q0EzQUYiLz4KPHN2ZyB4PSI4MCIgeT0iODAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMTIgMkM2LjQ4IDIgMiA2LjQ4IDIgMTJTNi40OCAyMiAxMiAyMlMyMiAxNy41MiAyMiAxMlMxNy41MiAyIDEyIDJaTTEzIDE3SDEwVjE1SDEzVjE3Wk0xMyAxM0gxMFY3SDEzVjEzWiIgZmlsbD0iIzZCNzI4MCIvPgo8L3N2Zz4KPC9zdmc+';
            }}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-slate-400">
            <div className="text-center">
              <div className="text-4xl">👜</div>
              <p className="text-xs uppercase tracking-wide">Preview not available</p>
            </div>
          </div>
        )}
        <div className="absolute right-3 top-3 rounded-full bg-white/80 px-3 py-1 text-xs font-semibold text-slate-600 shadow-sm">
          {product.department}
        </div>
      </div>

      <div className="flex flex-1 flex-col p-5">
        <div className="space-y-2">
          <h3 className="text-base font-semibold text-slate-900 line-clamp-2">
            {product.product_name}
          </h3>
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span className="rounded-full bg-blue-50 px-3 py-1 font-medium text-blue-600">
              {product.product_type}
            </span>
            <span>{product.color}</span>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-blue-600">
            <Eye className="h-4 w-4" />
            <span className="font-semibold">View Insights</span>
          </div>
          <div className="flex items-center gap-1 text-emerald-500">
            <TrendingUp className="h-4 w-4" />
            <span className="text-xs font-semibold">Ready</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductCard;
