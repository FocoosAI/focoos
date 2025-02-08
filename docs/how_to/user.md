# User Management

This section covers the steps to manage your user information on the FocoosAI platform.

## How to list your user info
To list your user info, the library provides a `get_user_info` function. This function will return a `User` object.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

user_info = focoos.get_user_info()

print(f"Email: {user_info.email}")
print(f"Created at: {user_info.created_at}")
print(f"Updated at: {user_info.updated_at}")
if user_info.company:
    print(f"Company: {user_info.company}")

print("\nQuotas:")
print(f"Total inferences: {user_info.quotas.total_inferences}")
print(f"Max inferences: {user_info.quotas.max_inferences}")
print(f"Used storage (GB): {user_info.quotas.used_storage_gb}")
print(f"Max storage (GB): {user_info.quotas.max_storage_gb}")
print(f"Active training jobs: {user_info.quotas.active_training_jobs}")
print(f"Max active training jobs: {user_info.quotas.max_active_training_jobs}")
print(f"Used MLG4DNXLarge training jobs hours: {user_info.quotas.used_mlg4dnxlarge_training_jobs_hours}")
print(f"Max MLG4DNXLarge training jobs hours: {user_info.quotas.max_mlg4dnxlarge_training_jobs_hours}")


```
`user_info` is a `User` object that contains information about the authenticated user, including their email address, company affiliation (if any), API key details, and platform usage quotas.

The User object contains the following fields:

- `email`: The email address associated with the user account
- `created_at`: Timestamp of when the user account was created
- `updated_at`: Timestamp of when the user account was last updated
- `company`: The company affiliation of the user (optional)
- `api_key`: The API key details used for authentication
- `quotas`: The platform usage quotas (see [`Quotas`](../../api/ports/#focoos.ports.Quotas)) allocated to the user

!!! note
    If you want to increase your quotas, please contact us via [support](mailto:info@focoos.ai).
