from django.contrib import admin

from .models import Expense


@admin.register(Expense)
class ExpenseAdmin(admin.ModelAdmin):
    list_display = ("title", "spent_on", "amount", "category", "payment", "reimbursable")
    radio_fields = {"payment": admin.HORIZONTAL}
