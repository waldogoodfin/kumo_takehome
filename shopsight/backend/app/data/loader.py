from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.config import get_settings


@dataclass
class DataBundle:
    articles: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame


class DataRepository:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._bundle: Optional[DataBundle] = None

    def load(self) -> DataBundle:
        if self._bundle is None:
            print("🔄 Loading H&M Business Data...")
            articles = pd.read_parquet(self._settings.articles_path)
            print(f"✅ Loaded {len(articles):,} products from H&M catalog")
            
            # Try to load real customer data, fallback to synthetic if missing
            try:
                customers = pd.read_parquet(self._settings.customers_path)
                print(f"✅ Loaded {len(customers):,} real customers from H&M data")
            except FileNotFoundError:
                print("⚠️  Customer data file not found, generating synthetic customers...")
                customers = self._generate_synthetic_customers()
                print(f"✅ Generated {len(customers):,} synthetic customers")
            
            transactions = self._generate_mock_transactions(articles, customers)
            print(f"✅ Generated {len(transactions):,} mock transactions")
            self._bundle = DataBundle(articles=articles, customers=customers, transactions=transactions)
        return self._bundle

    def _generate_synthetic_customers(self) -> pd.DataFrame:
        """Generate synthetic customer data when real H&M customer data is not available."""
        np.random.seed(42)  # For reproducible results
        
        num_customers = 50_000  # Reasonable number for demo
        customer_ids = [f"synthetic_{i:06d}" for i in range(num_customers)]
        
        # Generate realistic customer attributes
        ages = np.random.normal(35, 12, num_customers).clip(18, 80).astype(int)
        
        # Simple customer segments based on age
        segments = []
        for age in ages:
            if age < 25:
                segments.append("Gen Z")
            elif age < 40:
                segments.append("Millennial") 
            elif age < 55:
                segments.append("Gen X")
            else:
                segments.append("Boomer")
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'segment': segments,
            'synthetic': True  # Flag to indicate synthetic data
        })

    def refresh_transactions(self) -> None:
        if self._bundle is None:
            self.load()
        else:
            self._bundle.transactions = self._generate_mock_transactions(self._bundle.articles, self._bundle.customers)

    def _generate_mock_transactions(self, articles: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        settings = self._settings
        months = settings.transactions_months
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30 * months)

        # Use smaller sample to increase transaction density per product
        sample_products = articles.sample(n=min(3000, len(articles)), random_state=42)
        sample_customers = customers.sample(n=min(50_000, len(customers)), random_state=42)

        transactions = []
        transaction_id = 1
        current_date = start_date

        rng = np.random.default_rng(42)

        while current_date <= end_date:
            weekday_multiplier = 1.15 if current_date.weekday() < 5 else 0.85
            seasonal_multiplier = 1.3 if current_date.month in (11, 12) else 1.0
            base_volume = 60 * weekday_multiplier * seasonal_multiplier
            daily_transactions = int(rng.poisson(base_volume))

            chosen_products = sample_products.sample(n=min(daily_transactions, len(sample_products)), replace=True, random_state=transaction_id)
            chosen_customers = sample_customers.sample(n=min(daily_transactions, len(sample_customers)), replace=True, random_state=transaction_id)

            for product, customer in zip(chosen_products.itertuples(), chosen_customers.itertuples()):
                base_price = rng.uniform(15, 180)
                product_type = getattr(product, 'product_type_name', '').lower()
                if 'dress' in product_type:
                    base_price *= 1.4
                elif 'coat' in product_type:
                    base_price *= 1.8
                elif 'shirt' in product_type:
                    base_price *= 0.85

                quantity = rng.choice([1, 2, 3], p=[0.75, 0.2, 0.05])
                total_amount = base_price * quantity

                transactions.append(
                    {
                        'transaction_id': transaction_id,
                        'customer_id': getattr(customer, 'customer_id'),
                        'article_id': getattr(product, 'article_id'),
                        'transaction_date': datetime.combine(current_date, datetime.min.time()),
                        'quantity': quantity,
                        'unit_price': float(round(base_price, 2)),
                        'total_amount': float(round(total_amount, 2)),
                        'product_name': getattr(product, 'prod_name'),
                        'product_type': getattr(product, 'product_type_name'),
                        'department': getattr(product, 'department_name'),
                        'color': getattr(product, 'colour_group_name'),
                    }
                )
                transaction_id += 1

            current_date += timedelta(days=1)

        return pd.DataFrame(transactions)


def get_data_repository() -> DataRepository:
    return DataRepository()
