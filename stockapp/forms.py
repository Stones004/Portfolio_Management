from django import forms
from django.forms import formset_factory


class StockForm(forms.Form):
    symbol = forms.CharField(
        label="Stock symbol",
        max_length=30,
        widget=forms.TextInput(attrs={
            "placeholder": "e.g. INFY.NS",
            "class": "stock-input",
            "pattern": r"^[A-Za-z0-9.\-^=&]+$",
            "title": "Letters, numbers, periods and dashes only",
            "autocomplete": "off",
        }),
    )
    weight = forms.FloatField(
        label="Weight",
        min_value=0.0,
        error_messages={
            "required": "Please enter a weight.",
            "invalid": "Enter a valid number for weight.",
            "min_value": "Weight must be 0 or more.",
        },
        widget=forms.NumberInput(attrs={
            "placeholder": "e.g. 0.25 or 25",
            "step": "any",
            "min": "0",
            "class": "weight-input",
        }),
        help_text="Weights are rescaled to add up to 100%, so 0.25 or 25 both work.",
    )


StockFormSet = formset_factory(StockForm, extra=1, can_delete=True)
