import sqlite3
import random
from datetime import date, timedelta

DB_PATH = "data/sales.db"

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS sales;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS products;

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        industry TEXT,
        country TEXT,
        customer_size TEXT
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        category TEXT,
        cost REAL
    );

    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        date TEXT,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        discount REAL,
        region TEXT,
        sales_rep TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # --- Customers ---
    industries = ["Retail", "Tech", "Healthcare", "Manufacturing", "Finance"]
    sizes = ["Small", "Medium", "Large"]
    countries = ["USA", "UK", "Germany", "India", "Nepal"]

    customers = [
        (i, f"Customer {i}", random.choice(industries), random.choice(countries), random.choice(sizes))
        for i in range(1, 51)
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    # --- Products ---
    categories = ["Software", "Hardware", "Services"]
    products = [
        (i, f"Product {chr(64+i)}", random.choice(categories), round(random.uniform(20, 200), 2))
        for i in range(1, 11)
    ]
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    # --- Sales (with a deliberate August anomaly in West region, Product A) ---
    regions = ["North", "South", "East", "West"]
    reps = ["Alice", "Bob", "Charlie", "Diana", "Ethan"]

    sales = []
    sale_id = 1
    start_date = date(2026, 1, 1)

    for day_offset in range(243):  # Jan 1 to Aug 31
        current_date = start_date + timedelta(days=day_offset)
        num_orders = random.randint(15, 25)

        for _ in range(num_orders):
            customer_id = random.randint(1, 50)
            product_id = random.randint(1, 10)
            region = random.choice(regions)
            rep = random.choice(reps)
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(50, 300), 2)
            discount = round(random.uniform(0, 0.15), 2)

            # Inject anomaly: August + West + Product A => fewer orders
            if current_date.month == 8 and region == "West" and product_id == 1:
                if random.random() < 0.7:  # 70% chance to skip -> simulates order drop
                    continue

            sales.append((
                sale_id, current_date.isoformat(), customer_id, product_id,
                quantity, unit_price, discount, region, rep
            ))
            sale_id += 1

    cur.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        sales
    )

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH} with {len(customers)} customers, {len(products)} products, {len(sales)} sales.")


if __name__ == "__main__":
    create_database()