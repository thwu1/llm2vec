import pandas as pd

train_set = pd.read_csv("new_train_set.csv")
test_set = pd.read_csv("new_test_set.csv")

# print(train_set.head())
# print(test_set.head())

unique_id_name = train_set[['model_id', 'model_name']].drop_duplicates()
print(unique_id_name)

unique_id_name = test_set[['model_id', 'model_name']].drop_duplicates().reset_index(drop=True)
print(unique_id_name)

unique_id_name.to_csv("model_order.csv")