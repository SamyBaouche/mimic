from django.db import models


class Expense(models.Model):
    CATEGORIES = [
        ("travel", "Travel"),
        ("meals", "Meals"),
        ("office", "Office"),
        ("software", "Software"),
    ]
    PAYMENT = [("card", "Company card"), ("cash", "Cash"), ("transfer", "Bank transfer")]

    title = models.CharField(max_length=120)
    spent_on = models.DateField("date")
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=20, choices=CATEGORIES)
    payment = models.CharField(max_length=20, choices=PAYMENT, default="card")
    reimbursable = models.BooleanField(default=False)
    receipt = models.FileField(upload_to="receipts/", blank=True)

    def __str__(self):
        return self.title
