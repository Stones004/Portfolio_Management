from django import forms
from django.forms import formset_factory

class StockForm(forms.Form):
    symbol = forms.CharField(
        label='Stock Symbol',
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g. AAPL',
            'class': 'stock-input',
            'pattern': '^[A-Za-z0-9.\-]+$',  # Updated regex to allow dashes
            'title': 'Only letters, numbers, periods, and dashes allowed'
        })
    )

    weight = forms.FloatField(
        label='Weight',
        required=True,
        min_value=0.0,
        max_value=1.0,
        error_messages={
            'required': 'Please enter a weight.',
            'invalid': 'Enter a valid number for weight.',
            'min_value': 'Weight must be ≥ 0.',
            'max_value': 'Weight must be ≤ 1 (or normalized automatically).'
        },
        widget=forms.NumberInput(attrs={
            'placeholder': 'e.g. 0.25',
            'step': 'any',
            'min': '0',
            'max': '1',
            'class': 'weight-input'
        })
    )

StockFormSet = formset_factory(StockForm, extra=1,can_delete=True)
