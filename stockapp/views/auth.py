"""Account registration (login/logout use Django's built-in views, see urls.py)."""
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render


def register(request):
    if request.user.is_authenticated:
        return redirect("home")
    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)  # auto login after register
        return redirect("home")
    return render(request, "stockapp/register.html", {"form": form})
