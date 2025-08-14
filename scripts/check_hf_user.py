#!/usr/bin/env python3
"""
Check HuggingFace user information for the provided token
"""

from huggingface_hub import HfApi

def check_user(token: str):
    """Check which user the token belongs to"""
    api = HfApi(token=token)
    
    try:
        # Get user info
        user_info = api.whoami()
        
        print("🔍 HuggingFace Account Information:")
        print(f"  Username: {user_info.get('name', 'N/A')}")
        print(f"  Full Name: {user_info.get('fullname', 'N/A')}")
        print(f"  Email: {user_info.get('email', 'N/A')}")
        print(f"  Organizations: {user_info.get('orgs', [])}")
        
        # Check token permissions
        auth = user_info.get('auth', {})
        print(f"\n🔑 Token Permissions:")
        print(f"  Read access: {auth.get('accessToken', {}).get('read', False)}")
        print(f"  Write access: {auth.get('accessToken', {}).get('write', False)}")
        
        return user_info.get('name')
        
    except Exception as e:
        print(f"❌ Error checking user: {e}")
        return None

if __name__ == "__main__":
    import os
    token = os.getenv("HF_TOKEN", "your-hf-token-here")
    username = check_user(token)
    
    if username:
        print(f"\n✅ Your dataset should be uploaded to: {username}/khmer-medical-qa")