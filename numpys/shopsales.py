import numpy as np

rng = np.random.default_rng(seed=2)
quantity = rng.integers(low=0, high=80, size=(20, 7))
price = rng.integers(low=500, high=5000, size=20)
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# make prices a column so they match the table
price_col = price.reshape(20, 1)
product_rev = quantity * price_col

# totals
total_day_rev = np.sum(product_rev, axis=0)
total_product_rev = np.sum(product_rev, axis=1)
grand_total = np.sum(product_rev)

# best day and product
best_day = np.argmax(total_day_rev)
best_selling = np.argmax(total_product_rev) + 1

# 10% off products 5 to 10
discount_rev = product_rev.astype(float)
discount_rev[4:10] = discount_rev[4:10] * 0.9
discount_loss = grand_total - np.sum(discount_rev)

# cap units at 50
capped_quantity = np.minimum(quantity, 50)
capped_rev = capped_quantity * price_col
cap_loss = grand_total - np.sum(capped_rev)

print("Shapes:", quantity.shape, price.shape, "-> after reshape:", price_col.shape)
print("Daily revenue:")
for d, rev in zip(days, total_day_rev):
    print(f"  {d}: ₦{rev:,.0f}")
print(f"Grand total: ₦{grand_total:,.0f}")
print("Best day:", days[best_day])
print("Best-selling product:", best_selling)
print(f"Discount cost: ₦{discount_loss:,.0f} ({discount_loss / grand_total:.1%})")
print(f"Stock cap cost: ₦{cap_loss:,.0f} ({cap_loss / grand_total:.1%})")