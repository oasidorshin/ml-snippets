import pandas as pd


# Показыывать строки в ячейках таблицы полностью и автоматически переносить на новую строку
pd.set_option('display.max_colwidth', None)


# Макс количество строк/колонок
# min_rows нужно, чтобы показывало хотя бы n строк если больше чем макс
pd.set_option('display.min_rows', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', columns)


# Pivot table features example
# This creates (n_users, n_categories) table
pivot = pd.pivot_table(
    df,
    values='amount',
    index=['user_id'],
    columns=["category"],
    aggfunc={'amount': ["mean", "median"]},
    fill_value=0.0
    )

pivot.columns = ["_".join([str(el) for el in col_tuple]) for col_tuple in pivot.columns]
pivot = pivot.reset_index() # Optionally: pivot has user_id index, can reset it to add it to columns