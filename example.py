from mydcf import DCF_Valuation

# Assumptions to test
wacc = 0.11                 # 11% discount rate
growth_rate = 0.06          # 6% explicit growth
reinvestment_rate = 0.45    # 45% reinvestment
terminal_growth = 0.03      # 3% terminal growth
years = 5

model = DCF_Valuation("INFY.NS")

result = model.run_raw_dcf(
    wacc=wacc,
    growth_rate=growth_rate,
    reinvestment_rate=reinvestment_rate,
    terminal_growth=terminal_growth,
    years=years
)

print(f'\n\n\n\n\n\n\n\n\n')
print("Enterprise Value:", result["Enterprise Value"])
print("Equity Value:", result["Equity Value"])
print("Per Share Value:", result["Per Share Value"])

print("\nDCF Table:")
print(result["DCF Table"].round(0))