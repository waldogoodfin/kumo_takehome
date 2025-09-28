from fastapi import APIRouter

from app.schemas.dashboard import DashboardStats
from app.state import get_data_bundle

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def dashboard_stats() -> DashboardStats:
    bundle = get_data_bundle()
    transactions = bundle.transactions

    total_products = len(bundle.articles)
    total_customers = len(bundle.customers)
    total_transactions = len(transactions)
    total_revenue = float(transactions['total_amount'].sum())
    avg_order_value = float(transactions['total_amount'].mean()) if total_transactions else 0.0

    top = (
        transactions.groupby(['article_id', 'product_name'])
        .agg({'total_amount': 'sum', 'quantity': 'sum'})
        .sort_values('total_amount', ascending=False)
        .head(5)
        .reset_index()
    )

    top_products = [
        {
            'product_name': row['product_name'],
            'revenue': float(row['total_amount']),
            'units_sold': int(row['quantity']),
        }
        for _, row in top.iterrows()
    ]

    return DashboardStats(
        total_products=total_products,
        total_customers=total_customers,
        total_transactions=total_transactions,
        total_revenue=round(total_revenue, 2),
        avg_order_value=round(avg_order_value, 2),
        top_products=top_products,
    )
