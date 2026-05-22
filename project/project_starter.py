import ast
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from sqlalchemy import Engine, create_engine
from sqlalchemy.sql import text

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},
    # Product Types (priced per unit)
    {
        "item_name": "Paper plates",
        "category": "product",
        "unit_price": 0.10,
    },  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {
        "item_name": "Paper napkins",
        "category": "product",
        "unit_price": 0.02,
    },  # per napkin
    {
        "item_name": "Disposable cups",
        "category": "product",
        "unit_price": 0.10,
    },  # per cup
    {
        "item_name": "Table covers",
        "category": "product",
        "unit_price": 1.50,
    },  # per cover
    {
        "item_name": "Envelopes",
        "category": "product",
        "unit_price": 0.05,
    },  # per envelope
    {
        "item_name": "Sticky notes",
        "category": "product",
        "unit_price": 0.03,
    },  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {
        "item_name": "Invitation cards",
        "category": "product",
        "unit_price": 0.50,
    },  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {
        "item_name": "Party streamers",
        "category": "product",
        "unit_price": 0.05,
    },  # per roll
    {
        "item_name": "Decorative adhesive tape (washi tape)",
        "category": "product",
        "unit_price": 0.20,
    },  # per roll
    {
        "item_name": "Paper party bags",
        "category": "product",
        "unit_price": 0.25,
    },  # per bag
    {
        "item_name": "Name tags with lanyards",
        "category": "product",
        "unit_price": 0.75,
    },  # per tag
    {
        "item_name": "Presentation folders",
        "category": "product",
        "unit_price": 0.50,
    },  # per folder
    # Large-format items (priced per unit)
    {
        "item_name": "Large poster paper (24x36 inches)",
        "category": "large_format",
        "unit_price": 1.00,
    },
    {
        "item_name": "Rolls of banner paper (36-inch width)",
        "category": "large_format",
        "unit_price": 2.50,
    },
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(request_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        request_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :request_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"request_date": request_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


@dataclass
class ItemInfo:
    """Information about an item including its price and validity."""

    price: float | None
    is_valid: bool


def get_quote(item_names: List[str]) -> Dict[str, ItemInfo]:
    """
    Retrieve unit prices for the specified items from the inventory table.

    Args:
        item_names (List[str]): List of item names to look up prices for.

    Returns:
        Dict[str, ItemInfo]: A dictionary mapping item names to ItemInfo objects
                             containing their unit prices. `is_valid` is set to false
                             if the item does not exist in our database.
    """
    if not item_names:
        return {}

    # Build parameterized query for multiple items
    placeholders = ",".join([f":item_{i}" for i in range(len(item_names))])
    params = {f"item_{i}": name for i, name in enumerate(item_names)}

    query = f"""
        SELECT item_name, unit_price
        FROM inventory
        WHERE item_name IN ({placeholders})
    """

    result = pd.read_sql(query, db_engine, params=params)
    found_items = {
        row["item_name"]: float(row["unit_price"]) for _, row in result.iterrows()
    }

    # Build result with validity info for all requested items
    result_dict = {}
    for item_name in item_names:
        if item_name in found_items:
            result_dict[item_name] = ItemInfo(
                price=found_items[item_name], is_valid=True
            )
        else:
            result_dict[item_name] = ItemInfo(price=None, is_valid=False)

    return result_dict


def get_stock_level(item_name: str, request_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        request_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(request_date, datetime):
        request_date = request_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :request_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "request_date": request_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The  in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base."
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(str(item["item_name"]), as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * float(item["unit_price"])
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " OR ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
load_dotenv()

OPENAI_API_KEY = os.getenv("UDACITY_OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No OpenAI API key found.")

assert OPENAI_API_KEY is not None
if OPENAI_API_KEY.startswith("voc"):
    API_BASE = "https://openai.vocareum.com/v1"
else:
    raise ValueError("We do not support OpenAI keys here sorry.")


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


def is_valid_iso8601_date(date_string):
    """Check if a string is a valid ISO 8601 date in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# Tools for inventory agent
@tool
def check_inventory(request_date: str) -> str:
    """Retrieve the full inventory contents for the given date.

    Args:
        request_date: The date to check inventory for, in ISO format (YYYY-MM-DD).

    Returns:
        A human-readable summary of all item quantities in inventory as of the given date.

    """
    inventory: Dict[str, int] = get_all_inventory(request_date)
    if not is_valid_iso8601_date(request_date):
        return f"Invalid date format: '{request_date}'. Use YYYY-MM-DD."
    if not inventory:
        return f"No inventory data found as of '{request_date}'."
    items_str = "\n".join(
        f"  - {name}: {int(qty)} units" for name, qty in sorted(inventory.items())
    )
    return f"Inventory as of '{request_date}':\n{items_str}"


@tool
def check_stock_level(item_name: str, request_date: str) -> str:
    """Retrieve the stock level for a single inventory item on the given date.

    Args:
        item_name: The name of the item to look up.
        request_date: The date to check stock for, in ISO format (YYYY-MM-DD).

    Returns:
        A human-readable summary of the item's stock level as of the given date.

    """
    if not is_valid_iso8601_date(request_date):
        return f"Invalid date format: '{request_date}'. Use YYYY-MM-DD."
    result = get_stock_level(item_name, request_date)
    if result.empty or result.iloc[0]["current_stock"] == 0:
        return f"No stock found for '{item_name}' as of '{request_date}', or the item does not exist in our system."
    qty = int(result.iloc[0]["current_stock"])
    return f"Stock level for '{item_name}' as of '{request_date}': {qty} units."


# Tools for quoting agent


@tool
def get_item_prices(item_names: List[str]) -> str:
    """Retrieve unit prices for the specified items from the inventory.

    Args:
        item_names: A list of item names to look up prices for. The item names must exactly match
            their names in our system.

    Returns:
        A human-readable summary of item names and their unit prices.

    """
    if not item_names:
        return "No item names provided."
    prices = get_quote(item_names)
    lines = [f"Information for {len(prices)} item(s):"]
    for name in sorted(prices.keys()):
        item_info = prices[name]
        if item_info.is_valid:
            lines.append(f"  - {name}: ${item_info.price:.2f}")
        else:
            lines.append(f"  - {name}: Not a valid item in our inventory.")
    return "\n".join(lines)


@tool
def get_quote_history(search_terms: List[str]) -> str:
    """Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    results = search_quote_history(search_terms)
    if not results:
        return f"No historical quotes found matching: {', '.join(search_terms)}."
    lines = [
        f"Found {len(results)} historical quote(s) matching '{', '.join(search_terms)}':"
    ]
    for i, q in enumerate(results, 1):
        amount = (
            f"${q['total_amount']:.2f}" if q["total_amount"] >= 0 else "unavailable"
        )
        lines.append(
            f"\n[Quote {i}]"
            f"\n  order_date: {q['order_date']}"
            f"\n  Job type: {q['job_type']}, Order size: {q['order_size']}, Event: {q['event_type']}"
            f"\n  Total amount: {amount}"
            f"\n  Original request: {q['original_request']}"
            f"\n  Explanation: {q['quote_explanation']}"
        )
    result = "\n".join(lines)
    print(result)
    return result


# Tools for ordering agent
@tool
def get_delivery_timeline(order_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date based on the requested order quantity and a starting date.
    Inventory requires preparation before it can be delivered to the customer.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        order_date (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    delivery_date = get_supplier_delivery_date(order_date, quantity)
    return (
        f"For an order of {quantity} unit(s) placed on {order_date}, "
        f"the estimated delivery date is {delivery_date}."
    )


@tool
def fulfill_order(
    item_name: str, quantity: int, price: float, transaction_date: str
) -> str:
    """Creates and finalises a single sale for that customer.

    Args:
        item_name: The name of the item being ordered.
        quantity: The number of units in the order.
        price: The total price of the order.
        transaction_date: The date of the transaction in ISO format (YYYY-MM-DD).

    Returns:
        A confirmation message including the transaction ID of the fulfilled order.
    """
    try:
        transaction_id = create_transaction(
            item_name, "sales", quantity, price, transaction_date
        )
        return f"Order successfully fulfilled for {quantity} unit(s) of '{item_name}' at ${price:.2f} total on {transaction_date}. Transaction ID: {transaction_id}."
    except Exception as e:
        return f"Failed to fulfill order: {str(e)}"


class OrchestrationAgent(ToolCallingAgent):
    def __init__(self, model, managed_agents):
        instructions = """You are a helpful customer service agent for the "Beaver's Choice" paper company.
You are the orchestration agent for other agents (i.e. team members).
The request date MUST always be included when delegating tasks to team members.
Canonical product names should be used in interaction with team members rather than the customer's product names.
Before any other interactions with team members, ask the inventory team member to convert the customer's product names
into canonical product names.
Unless specified otherwise, assume all inventory items follow U.S. paper size conventions.
If the product is not available in our inventory, do not finalize a sale but rather repond politely to the customer
with a list of these alternatives.

You can generate quotes for customers.
You can finalise sales.
When a customer requests paper, they mean that they would like a sale finalised.
When checking inventory for multiple items, check all items in a single request to the inventory team member.
Inventory requires preparation time before it can be delivered; ask the "quote" team member to calculate these so these
can be provided to the customer."""

        super().__init__(
            model=model,
            tools=[],
            name="orchestrator",
            description="",  # Not used for the top-level agent.
            managed_agents=managed_agents,
            instructions=instructions,
        )


class InventoryAgent(ToolCallingAgent):
    def __init__(self, model, tools):
        instructions = """You are an inventory team member for the "Beaver's Choice" paper company.
You can answer queries about inventory levels.
Many customer queries will not include canonical product names. To find canonical product names, search through all inventory
contents for a similar product. When replying with canonical product names, reply with the customer's name for each product and it's canonical name.
Unless stated otherwise in a product name, assume all products follow U.S. paper size conventions.
If a customer's product name is not a product we have, include a comment about this in the reply and report back with suggestions for similar products.
If no request date is supplied, respond with a complaint about that.
Product names (item names) are case sensitive.
"""
        super().__init__(
            model=model,
            tools=tools,
            name="inventory",
            description="""This team member can answer queries about inventory contents and inventory levels. This team member can attempt to
            match product names with inventory contents. A request date MUST be supplied with all requests.""",
            instructions=instructions,
        )


class QuoteAgent(ToolCallingAgent):
    def __init__(self, model, tools):
        super().__init__(
            model=model,
            tools=tools,
            name="quote",
            description="This team member generates quotes for products based on inventory availability. It requires exact product names from our inventory.",
            instructions="""You are a quote managing team member for the "Beaver's Choice" paper company.
You are responsible for generating quotes for products based on inventory availability.
If no request date is supplied, respond with a complaint about that.
Product names (item names) are case sensitive.
""",
        )


class SalesFinalisationAgent(ToolCallingAgent):
    def __init__(self, model, tools):
        super().__init__(
            model=model,
            tools=tools,
            name="sales_finalisation",
            description="This team member finalises sales orders by updating inventory and generating invoices.",
            instructions="""You are a sales team member for the "Beaver's Choice" paper company.
You are responsible for finalising sales orders.
If no request date is supplied, respond with a complaint about that.
You do sales in two steps:
    1. You first verify the exact names of products using the check_inventory tool. If
       the product does not exist in our inventory, do not proceed with the sale but reply
       with an error saying that the given product could not be found.
    2. You then finalise the sale using the fulfill_order tool.
All products in the sale must have their exact name in our inventory.
Product names (item names) are case sensitive.""",
        )


# Run your test scenarios by writing them here. Make sure to keep track of them.


def create_agent() -> ToolCallingAgent:
    # Initialize agents
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        api_base=API_BASE,
    )

    inventory_agent = InventoryAgent(
        model=model, tools=[check_inventory, check_stock_level]
    )
    quote_agent = QuoteAgent(
        model=model,
        tools=[get_quote_history, get_item_prices, get_delivery_timeline],
    )
    sales_finalisation_agent = SalesFinalisationAgent(
        model=model, tools=[check_inventory, fulfill_order]
    )

    orchestrator: ToolCallingAgent = OrchestrationAgent(
        model=model,
        managed_agents=[inventory_agent, quote_agent, sales_finalisation_agent],
    )
    return orchestrator


def run_one():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests = pd.read_csv("quote_requests.csv")
        quote_requests["request_date"] = pd.to_datetime(
            quote_requests["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests.dropna(subset=["request_date"], inplace=True)
        quote_requests = quote_requests.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    agent: ToolCallingAgent = create_agent()

    job = "office manager"
    need_size = "small"
    event = "ceremony"
    request = """I would like to request the following paper supplies for the ceremony:

- 200 sheets of A4 glossy paper
- 100 sheets of heavy cardstock (white)
- 100 sheets of colored paper (assorted colors)

I need these supplies delivered by April 15, 2025. Thank you."""
    request_date = "2025-04-01"

    print(f"Context: {job} organizing {event}")
    print(f"Request Date: {request_date}")
    print(f"Cash Balance: ${current_cash:.2f}")
    print(f"Inventory Value: ${current_inventory:.2f}")

    request_with_date = f"""
    Request Date (YYYY-MM-DD): {request_date}
    Customer role: {job}
    Job size: {need_size}
    Event type: {event}
    --
    {request}"""
    response = agent.run(request_with_date)

    # Update state
    report = generate_financial_report(request_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    print(f"Response: {response}")
    print(f"Updated Cash: ${current_cash:.2f}")
    print(f"Updated Inventory: ${current_inventory:.2f}")

    # Final report
    final_date = quote_requests["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")


def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests = pd.read_csv("quote_requests.csv")
        quote_requests["request_date"] = pd.to_datetime(
            quote_requests["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests.dropna(subset=["request_date"], inplace=True)
        quote_requests = quote_requests.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    agent: ToolCallingAgent = create_agent()

    results = []
    for i, (idx, row) in enumerate(quote_requests.iterrows()):
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {i + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        job = row["job"]
        event = row["event"]
        need_size = row["need_size"]
        request = row["request"]
        request_with_date = f"""
Date of request (YYYY-MM-DD): {request_date}
Customer role: {job}
Job size: {need_size}
Event type: {event}
--
{request}"""
        try:
            response = agent.run(request_with_date)
        except Exception as e:
            print(f"ERROR processing request {i + 1}: {e}")
            response = f"Error: {str(e)}"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": i + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    # run_one()
    run_test_scenarios()
