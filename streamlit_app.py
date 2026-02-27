"""Streamlit Cloud entry point - redirects to dashboard app."""

# Import and run the dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()
else:
    # When imported by Streamlit, run main directly
    main()
