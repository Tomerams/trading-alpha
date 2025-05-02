from datetime import date, timedelta
from pydantic import BaseModel, Field


class UpdateIndicatorsData(BaseModel):
    stock_ticker: str = "QQQ"
    start_date: str = Field(
        default_factory=lambda: (date.today() - timedelta(days=365)).isoformat()
    )
    end_date: str = Field(
        default_factory=lambda: (date.today() - timedelta(days=1)).isoformat()
    )
