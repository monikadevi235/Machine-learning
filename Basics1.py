customer_ids = ('C1', 'C2', 'C3')

customers = {
    'C1': {'name': 'Anu', 'amount': 1200},
    'C2': {'name': 'Bala', 'amount': 700},
    'C3': {'name': 'Chitra', 'amount': 300}
}


def calculate_discount(amount):
    if amount >= 1000:
        return amount * 0.20
    elif amount >= 500:
        return amount * 0.10
    else:
        return 0


print("SHOPPING BILL REPORT")
print("-" * 40)

for cid in customer_ids:
    data = customers[cid]

    discount = calculate_discount(data['amount'])
    final_amount = data['amount'] - discount

    print("Customer ID   :", cid)
    print("Name          :", data['name'])
    print("Total Amount  :", data['amount'])
    print("Discount      :", discount)
    print("Payable Amt   :", final_amount)
    print("-" * 40)
