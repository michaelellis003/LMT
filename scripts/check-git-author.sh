#!/bin/sh
# Pre-commit hook: block commits from noreply email addresses.
# This prevents phantom contributors showing up on GitHub.
email=$(git config user.email)
case "$email" in
  *noreply*)
    echo "ERROR: git user.email contains 'noreply' ($email)"
    echo "Fix with: git config user.email YOUR_REAL_EMAIL"
    exit 1
    ;;
esac
