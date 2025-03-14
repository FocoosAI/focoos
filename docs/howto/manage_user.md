# User Management

Managing your user information is essential for tracking account details, platform usage, and resource quotas.
The Focoos library provides built-in methods to retrieve your user information, including email, API key details, company affiliation, and allocated usage quotas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/user_info.ipynb)

In this guide, we will cover the following topics:

1. [📄 Retrieve User Information](#retrieve-user-information)
2. [📊 Monitor Quota Usage](#monitor-your-quota-usage)


## Retrieve user information
To access your user details, you can use the `get_user_info` function provided by the Focoos library. This function returns a `User` object containing key account information.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

user_info = focoos.get_user_info()

print(f"Email: {user_info.email}")
print(f"Created at: {user_info.created_at}")
print(f"Updated at: {user_info.updated_at}")
if user_info.company:
    print(f"Company: {user_info.company}")

```
The `user_info` object contains the following fields:

- `email`: The email address associated with your account
- `created_at`: Timestamp of when the account was created
- `updated_at`: Timestamp of the last account update
- `company`: The company affiliated with your account (if applicable)
- `api_key`: Your API key details used for authentication
- `quotas`: Your allocated platform usage quotas (see [`Quotas`](/focoos/api/ports/#focoos.ports.Quotas) allocated to the user).


## Monitor your quota usage

The Focoos platform enforces usage quotas to manage resources efficiently.
You can retrieve your current quota limits using the `get_user_info` function like this:


```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

user_info = focoos.get_user_info()

print("\nQuotas:")
print(f"Total inferences: {user_info.quotas.total_inferences}")
print(f"Max inferences: {user_info.quotas.max_inferences}")
print(f"Used storage (GB): {user_info.quotas.used_storage_gb}")
print(f"Max storage (GB): {user_info.quotas.max_storage_gb}")
print(f"Active training jobs: {user_info.quotas.active_training_jobs}")
print(f"Max active training jobs: {user_info.quotas.max_active_training_jobs}")
print(f"Used training jobs hours: {user_info.quotas.used_mlg4dnxlarge_training_jobs_hours}")
print(f"Max training jobs hours: {user_info.quotas.max_mlg4dnxlarge_training_jobs_hours}")

```

!!! note
    If you need to increase your quotas, please contact us at [support](mailto:support@focoos.ai).
