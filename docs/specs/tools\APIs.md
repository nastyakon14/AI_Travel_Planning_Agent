# Контракт search_flights

```
Input: origin (str), destination (str), departure_date (YYYY-MM-DD), return_date (YYYY-MM-DD).
Output: Array of {flight_id, airline, price_eur, time}.
```

- Защита (Side effects & Guardrails): Тулы обернуты в try-except. При таймауте (>5 сек) тул возвращает строку "Error: API timeout. Proceed without real-time prices." Агент обучен воспринимать эту строку и использовать mock-данные.
